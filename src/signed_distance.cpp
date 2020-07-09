#include <sdf_support.h>
#include <lpt_model.h>
#include <sys_platform.h>

///////////////////////////////////////////////////////////////////////////////

#include <evaluators/eval_precalc_threaded.h>

#include <evaluators/eval_grid4_threaded.h>
#include <evaluators/eval_grid8_threaded.h>
#include <evaluators/eval_grid16_threaded.h>


///////////////////////////////////////////////////////////////////////////////

//note: random SIMD references
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/
// https://msdn.microsoft.com/en-us/library/hh977023.aspx
// https://db.in.tum.de/~finis/x86-intrin-cheatsheet-v2.1.pdf
// https://www.inf.ethz.ch/personal/markusp/teaching/263-2300-ETH-spring11/slides/class17.pdf
//
// https://msdn.microsoft.com//en-us/library/26td21ds.aspx
//
// https://www.agner.org/optimize/vectorclass.pdf
// https://www.agner.org/optimize/optimizing_cpp.pdf
// https://www.agner.org/optimize/instruction_tables.pdf
//
// https://deplinenoise.files.wordpress.com/2015/03/gdc2015_afredriksson_simd.pdf
// http://www.cs.uu.nl/docs/vakken/magr/2017-2018/files/SIMD%20Tutorial.pdf


// ====
void init_sdf( sdf_t *sdf, aabb_t bb, int32_t siz_x, int32_t siz_y, int32_t siz_z, int32_t simd_siz )
{
	//enum { SIMD_SIZ = 16, SIMD_ALIGN=4*SIMD_SIZ };
	const int32_t simd_align = sizeof(float32_t) * simd_siz;

	sdf->header.dim_x = siz_x;
	sdf->header.dim_y = siz_y;
	sdf->header.dim_z = siz_z;

	//note: extend bb by a border
	const int32_t border_siz = sdf->header.dim_x/4;
	const vec3_t bb_gridcell_siz = (bb.mx - bb.mn) * vec3_t( 1.0f/sdf->header.dim_x, 1.0f/sdf->header.dim_y, 1.0f/sdf->header.dim_z );
	bb.mn = bb.mn - (float32_t)border_siz * bb_gridcell_siz;
	bb.mx = bb.mx + (float32_t)border_siz * bb_gridcell_siz;

	sdf->header.bb_mn_x = bb.mn.x;
	sdf->header.bb_mn_y = bb.mn.y;
	sdf->header.bb_mn_z = bb.mn.z;

	sdf->header.bb_mx_x = bb.mx.x;
	sdf->header.bb_mx_y = bb.mx.y;
	sdf->header.bb_mx_z = bb.mx.z;

	sdf->data = (float32_t*)_aligned_malloc( sizeof(float32_t) * sdf->header.dim_x * sdf->header.dim_y * sdf->header.dim_z, simd_align );

	#ifndef NDEBUG
	sdf->eval_points = (vec3_t*)_aligned_malloc( 3*sizeof(float32_t) * sdf->header.dim_x * sdf->header.dim_y * sdf->header.dim_z, simd_align );
	#endif //NDEBUG
}

// ====
void deinit_sdf( sdf_t *sdf )
{
	_aligned_free( sdf->data );

	#ifndef NDEBUG
	_aligned_free( sdf->eval_points );
	#endif
}


// ==============================================================

bool equals_sdf( sdf_t const * const s01, sdf_t const * const s02 )
{
	assert( s01 != nullptr && s01->data != nullptr );
	assert( s02 != nullptr && s02->data != nullptr );
	
	const int32_t count = s01->header.dim_x * s01->header.dim_y * s01->header.dim_z;
	assert( count == s02->header.dim_x * s02->header.dim_y * s02->header.dim_z );
	return memcmp( s01->data, s02->data, sizeof(float32_t) * count ) == 0;
}

bool equals_approximately_sdf( sdf_t const * const s01, sdf_t const * const s02 )
{
		bool equals = true;

		assert( s01->header.dim_x == s02->header.dim_x );
		assert( s01->header.dim_y == s02->header.dim_y );
		assert( s01->header.dim_z == s02->header.dim_z );

		float32_t mindiff =  FLT_MAX;
		float32_t maxdiff = -FLT_MAX;
		const int32_t count = s01->header.dim_x * s01->header.dim_y * s01->header.dim_z;
		for ( int i=0,n=count; i<n; ++i )
		{
			const float32_t d0 = s01->data[ i ];
			const float32_t d1 = s02->data[ i ];
			if ( !AlmostEqualRelative( d0, d1) )
			{
				equals = false;
				//printf("%#010x != %#010x\n", *(uint32_t*)&d0, *(uint32_t*)&d1);
			}

			float32_t diff = abs( d1 - d0 );
			if ( diff < mindiff )
				mindiff = diff;
			if ( diff > maxdiff )
				maxdiff = diff;
		}

		//if( mindiff < maxdiff ) printf( "delta: [min;max]=[%f;%f]\n", mindiff, maxdiff );
		if( mindiff < maxdiff ) printf( "delta: [min;max]=[%.10e;%.10e]\n", mindiff, maxdiff );

		return equals;
}

// ==============================================================

int main( int argc, char *argv[] )
{
	setvbuf(stdout, NULL, _IONBF, 0);

    #ifndef NDEBUG
    printf( "==================\n===== DEBUG ======\n==================\n\n");
    #endif

	printf( "CPU: \"%s\"\n", InstructionSet::Brand().c_str() );

	printf( "supported extensions: ");
	print_support( true );
	
	printf( "\n(unsupported extensions: ");
	print_support( false );
	printf( "\n");

	const bool path_avx256 = InstructionSet::AVX2();
	const bool path_avx512 = InstructionSet::AVX512F();

	cpuinfo_t cpuinfo = calc_num_cores();

	bool maxload = false;
	if ( argc > 1 && std::string(argv[1]) == std::string("--maxload") )
	{
		maxload = true;
		printf("maxload, %d threads\n", cpuinfo.num_cores_logical);
	}

	init_timers();
	init_profiler();
	atexit( deinit_profiler );

	PROFILE_THREADNAME( "mainthread" );

	PROFILE_FUNC();

	const int test_siz = 64;
	const int maxload_siz = 128;
	const int GRID_SIZ_X = maxload ? maxload_siz : test_siz;
	const int GRID_SIZ_Y = maxload ? maxload_siz : test_siz;
	const int GRID_SIZ_Z = maxload ? maxload_siz : test_siz;

	//note: assimp import to trimesh
	PROFILE_ENTER("loadmodel");
	//indexed_triangle_mesh_t const * const mesh = lpt::loadmodel_assimp__posonly( "../data/sphere.obj" );
	//indexed_triangle_mesh_t const * const mesh = lpt::loadmodel_assimp__posonly( "../data/pyramid_blob.obj" );
	//indexed_triangle_mesh_t const * const mesh = lpt::loadmodel_assimp__posonly( "../data/tetra_nonormals.obj" );
	indexed_triangle_mesh_t const * const mesh = lpt::loadmodel_assimp__posonly( "../data/bunny.obj" );
	//indexed_triangle_mesh_t const * const mesh = lpt::loadmodel_assimp__posonly( "../data/tigre_sumatra_sketchfab.obj" );
	PROFILE_LEAVE("loadmodel");

	assert( mesh->positions.size() % 3 == 0 );
	vec3_t const * const positions = reinterpret_cast<vec3_t const * const>( &mesh->positions[0] );

	aabb_t bb;
	for ( size_t i=0, in=mesh->positions.size()/3; i<in; ++i )
	{
		const vec3_t p = positions[ i ];
		bb.mn = min( bb.mn, p );
		bb.mx = max( bb.mx, p );
	}

	printf( "tris: %d\ngrid(%d,%d,%d)\n", (int)mesh->tri_indices.size()/3, GRID_SIZ_X, GRID_SIZ_Y, GRID_SIZ_Z );

	sdf_t sdf;
	sdf_t sdf_simd;

	enum INSTR : int { SCALAR=0, SSE, AVX256, AVX512, INSTRCOUNT };
	int instr = maxload ? INSTR::AVX512 : INSTR::SCALAR;
	while ( instr >= 0 && instr < INSTR::INSTRCOUNT)
	{
		PROFILE_SCOPE("sdf_calc");

		if ( instr == INSTR::SCALAR )
		{
			printf( "\nScalar(single, ms):\n" );
			init_sdf( &sdf, bb, GRID_SIZ_X, GRID_SIZ_Y, GRID_SIZ_Z, 1 );
			int i = maxload ? cpuinfo.num_cores_logical : 1;
			for ( int n=cpuinfo.num_cores_logical; i<=n; ++i )
			{
				const uint64_t t0 = gettime_ms();
				eval_sdf__precalc_threaded( sdf, mesh, i );
				const uint64_t t1 = gettime_ms();
				printf( "%d\n", (int)(t1-t0) );
			}
		}

		//note: path_sse... assumed universally available...
		if ( instr == INSTR::SSE)
		{
			printf("\nSSE(4-wide, ms):\n");
			init_sdf(&sdf_simd, bb, GRID_SIZ_X, GRID_SIZ_Y, GRID_SIZ_Z, 4);

			#if (USE_PROFILER == PROFILER_MICROPROFILE )
			MICROPROFILE_TIMELINE_ENTER_STATIC(MP_POWDERBLUE, "SSE");
			#endif
			int i = maxload ? cpuinfo.num_cores_logical : 1;
			for ( int n = cpuinfo.num_cores_logical; i <= n; ++i)
			{
				const uint64_t t0 = gettime_ms();
				eval_sdf__grid4_threaded(sdf_simd, mesh, i);
				const uint64_t t1 = gettime_ms();
				printf("%d\n", (int)(t1 - t0));
			}

			#if (USE_PROFILER == PROFILER_MICROPROFILE )
			MICROPROFILE_TIMELINE_LEAVE_STATIC("SSE");
			#endif

			if (!maxload)
			{
				printf( "scalar == sse: %s\n", equals_sdf( &sdf, &sdf_simd) ? "true" : "false" );
				printf( "scalar ~= sse: %s\n", equals_approximately_sdf( &sdf, &sdf_simd) ? "true" : "false" );
			}
		}

		if ( instr == INSTR::AVX256 && path_avx256 )
		{
			printf( "\nAVX2(8-wide, ms):\n" );
			init_sdf( &sdf_simd, bb, GRID_SIZ_X, GRID_SIZ_Y, GRID_SIZ_Z, 8 );

			#if (USE_PROFILER == PROFILER_MICROPROFILE )
			MICROPROFILE_TIMELINE_ENTER_STATIC(MP_CHARTREUSE, "AVX256");
			#endif
			int i = maxload ? cpuinfo.num_cores_logical : 1;
			for ( int n=cpuinfo.num_cores_logical; i<=n; ++i )
			{
				const uint64_t t0 = gettime_ms();
				eval_sdf__grid8_threaded( sdf_simd, mesh, i );
				const uint64_t t1 = gettime_ms();
				printf( "%d\n", (int)(t1-t0) );
			}

			#if (USE_PROFILER == PROFILER_MICROPROFILE )
			MICROPROFILE_TIMELINE_LEAVE_STATIC("AVX256");
			#endif

			if ( !maxload )
			{
				printf( "sse == avx256: %s\n", equals_sdf( &sdf, &sdf_simd) ? "true" : "false" );
				printf( "sse ~= avx256: %s\n", equals_approximately_sdf( &sdf, &sdf_simd ) ? "true" : "false" );
			}
		}

		if ( instr == INSTR::AVX512 && path_avx512 )
		{
			printf("\nAVX512(16-wide, ms):\n");
			init_sdf(&sdf_simd, bb, GRID_SIZ_X, GRID_SIZ_Y, GRID_SIZ_Z, 16);

			#if (USE_PROFILER == PROFILER_MICROPROFILE )
			MICROPROFILE_TIMELINE_ENTER_STATIC(MP_SALMON, "AVX512");
			#endif
			int i = maxload ? cpuinfo.num_cores_logical : 1;
			for ( int n = cpuinfo.num_cores_logical; i <= n; ++i)
			{
				const uint64_t t0 = gettime_ms();
				eval_sdf__grid16_threaded(sdf_simd, mesh, i);
				const uint64_t t1 = gettime_ms();
				printf("%d\n", (int)(t1 - t0));
			}
			#if (USE_PROFILER == PROFILER_MICROPROFILE )
			MICROPROFILE_TIMELINE_LEAVE_STATIC("AVX512");
			#endif

			if (!maxload )
			{
				printf("sse == avx512: %s\n", equals_sdf(&sdf, &sdf_simd) ? "true" : "false");
				printf("sse ~= avx512: %s\n", equals_approximately_sdf(&sdf, &sdf_simd) ? "true" : "false");
			}
		}

		instr = maxload ? --instr : ++instr;

	} //while instr
	
	delete mesh;

	//{
	//	FILE *f;
	//	fopen_s( &f, "sdf.bin", "wb" ); //TODO: write to <filename>_128x128x128_sdf.bin
	//	fwrite( &sdf.header, sizeof(header_t), 1, f );
	//	fwrite( sdf.data, sizeof(float32_t) * sdf.header.dim_x * sdf.header.dim_y * sdf.header.dim_z, 1, f );
	//	fclose( f );
	//}

	deinit_sdf( &sdf );

	return 0;
}

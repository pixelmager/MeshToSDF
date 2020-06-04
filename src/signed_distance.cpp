#include <sdf_support.h>
#include <lpt_model.h>

#include <sys_platform.h>

///////////////////////////////////////////////////////////////////////////////

#include <evaluators/eval_bruteforce.h>
#include <evaluators/eval_precalc.h>

#include <evaluators/eval_vectorsimd.h>
#include <evaluators/eval_vectorsimd_threaded.h>

#include <evaluators/eval_grid4.h>
#include <evaluators/eval_grid8.h>
#include <evaluators/eval_grid16.h>

#include <evaluators/eval_tris4.h>
#include <evaluators/eval_tris8.h>
#include <evaluators/eval_tris16.h>

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

using namespace lpt;

// ====
void init_sdf( sdf_t *sdf, aabb_t bb, int32_t siz_x, int32_t siz_y, int32_t siz_z )
{
	enum { SIMD_SIZ = 16, SIMD_ALIGN=4*SIMD_SIZ };

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

	sdf->data = (float32_t*)_aligned_malloc( sizeof(float32_t) * sdf->header.dim_x * sdf->header.dim_y * sdf->header.dim_z, SIMD_ALIGN );

	#ifndef NDEBUG
	sdf->eval_points = (vec3_t*)_aligned_malloc( 3*sizeof(float32_t) * sdf->header.dim_x * sdf->header.dim_y * sdf->header.dim_z, SIMD_ALIGN );
	#endif //NDEBUG
}

void print_support( bool print_matching )
{
	#define LPT_FNC(M) InstructionSet::M()
	#define LPT_DOPRINT(M) if(print_matching == LPT_FNC(M) ) printf( #M ## " " )
	LPT_DOPRINT( SSE );
	LPT_DOPRINT( SSE2 );
	LPT_DOPRINT( SSE3 );
	LPT_DOPRINT( SSE41 );
	LPT_DOPRINT( SSE42 );
	LPT_DOPRINT( SSE4a );
	LPT_DOPRINT( SSSE3 );

	LPT_DOPRINT( AVX );
	LPT_DOPRINT( AVX2 );
	LPT_DOPRINT( AVX512CD );
	LPT_DOPRINT( AVX512ER );
	LPT_DOPRINT( AVX512F );
	LPT_DOPRINT( AVX512PF );

	LPT_DOPRINT( FMA );
	#undef LPT_FNC
	#undef LPT_DOPRINT
}

//TODO: precalc angle-weighted normals
//      https://github.com/janba/GEL/blob/master/src/demo/MeshDistance/meshdist.cpp
int main()
{
	printf( "supported extensions: ");
	print_support( true );
	
	printf( "\n(unsupported extensions: ");
	print_support( false );
	printf( "\n");

	bool path_avx256 = InstructionSet::AVX2();
	bool path_avx512 = InstructionSet::AVX512F();

	//system("pause");

	init_timers();
	init_profiler();
	atexit( deinit_profiler );

	PROFILE_THREADNAME( "mainthread" );

	PROFILE_FUNC();

	//enum { GRID_SIZ_X = 25, GRID_SIZ_Y = 25, GRID_SIZ_Z = 25 };
	//enum { GRID_SIZ_X = 50, GRID_SIZ_Y = 50, GRID_SIZ_Z = 50 };
	//enum { GRID_SIZ_X = 100, GRID_SIZ_Y = 100, GRID_SIZ_Z = 100 };
	//
	//enum { GRID_SIZ_X = 16, GRID_SIZ_Y = 16, GRID_SIZ_Z = 16 };
	//enum { GRID_SIZ_X = 32, GRID_SIZ_Y = 32, GRID_SIZ_Z = 32 };
	//enum { GRID_SIZ_X = 64, GRID_SIZ_Y = 64, GRID_SIZ_Z = 64 };
	enum { GRID_SIZ_X = 128, GRID_SIZ_Y = 128, GRID_SIZ_Z = 128 };

	//note: assimp import to trimesh
	PROFILE_ENTER("loadmodel");
	//indexed_triangle_mesh_t const * const mesh = lpt::loadmodel_assimp__posonly( "../data/sphere.obj" );
	//indexed_triangle_mesh_t const * const mesh = lpt::loadmodel_assimp__posonly( "../data/pyramid_blob.obj" );
	//indexed_triangle_mesh_t const * const mesh = lpt::loadmodel_assimp__posonly( "../data/tetra_nonormals.obj" );
	indexed_triangle_mesh_t const * const mesh = lpt::loadmodel_assimp__posonly( "../data/bunny.obj" );
	//indexed_triangle_mesh_t const * const mesh = lpt::loadmodel_assimp__posonly( "../data/tigre_sumatra_sketchfab.obj" );
	//indexed_triangle_mesh_t const * const mesh = lpt::loadmodel_assimp__posonly( "../data/DeformedPigs.fbx" );
	PROFILE_LEAVE("loadmodel");

	assert( mesh->positions.size() % 3 == 0 );
	vec3_t const * const positions = reinterpret_cast<vec3_t const * const>( &mesh->positions[0] );

	//TODO: would it be an optimisation to convert indexed trimesh to pure triangle-array (would map better to simd?)
	//TODO: build one-level grid of active cells

	aabb_t bb;
	for ( size_t i=0, in=mesh->positions.size()/3; i<in; ++i )
	{
		const vec3_t p = positions[ i ];
		bb.mn = min( bb.mn, p );
		bb.mx = max( bb.mx, p );
	}

	sdf_t sdf;
	init_sdf( &sdf, bb, GRID_SIZ_X, GRID_SIZ_Y, GRID_SIZ_Z );

	printf( "tris: %d\ngrid(%d,%d,%d)\n", (int)mesh->tri_indices.size()/3, sdf.header.dim_x, sdf.header.dim_y, sdf.header.dim_z );

	//#define DO_MULTIPLE_TIMINGS
	#if defined ( DO_MULTIPLE_TIMINGS )
	enum { NUM_ITER=7 };
	std::vector<uint64_t> timings( NUM_ITER );
	uint64_t t_min = INT_MAX;
	uint64_t t_max = 0;
	for ( int i=0,n=NUM_ITER; i<n; ++i )
	#endif //DO_MULTIPLE_TIMINGS

	{
		PROFILE_SCOPE("sdf_calc");

		const uint64_t t0_ms = gettime_ms();
		
		//eval_sdf__bruteforce( sdf, mesh );
		//eval_sdf__precalc (sdf, mesh );

		//eval_sdf__precalc_simd_aos( sdf, mesh );

		//eval_sdf__precalc_simd_4grid( sdf, mesh );
		//eval_sdf__precalc_simd_8grid( sdf, mesh );
		//eval_sdf__precalc_simd_16grid( sdf, mesh );

        if ( path_avx512 )
        {
            const uint64_t t0 = gettime_ms();
            eval_sdf__grid16_threaded( sdf, mesh );
            const uint64_t t1 = gettime_ms();
            printf( "AVX512: %dms\n", (int)(t1-t0) );
        }

        if ( path_avx256 )
        {
            const uint64_t t0 = gettime_ms();
            eval_sdf__grid8_threaded(sdf, mesh);
            const uint64_t t1 = gettime_ms();
            printf( "AVX2(256): %dms\n", (int)(t1-t0) );
        }

        {
            const uint64_t t0 = gettime_ms();
            eval_sdf__grid4_threaded(sdf, mesh);
            const uint64_t t1 = gettime_ms();
            printf( "SSE(128) %dms\n", (int)(t1-t0) );
        }

        {
            const uint64_t t0 = gettime_ms();
            eval_sdf__precalc_threaded(sdf, mesh);
            const uint64_t t1 = gettime_ms();
            printf( "scalar %dms\n", (int)(t1-t0) );
        }
		
        ////eval_sdf__aos_threaded( sdf, mesh );
		//
		////eval_sdf__simd_soa_4tris( sdf, mesh );
		////eval_sdf__simd_soa_8tris( sdf, mesh );
		////eval_sdf__simd_soa_16tris( sdf, mesh );
		//
		//const uint64_t t1_ms = gettime_ms();
		//
		//#if defined ( DO_MULTIPLE_TIMINGS )
		//timings[i] = t1_ms - t0_ms;
		//t_min = std::min( t_min, timings[i] );
		//t_max = std::max( t_max, timings[i] );
		////printf( "timing %d: sdf %dms\n", i, (int)timings[i] );
		//#else
		//printf( "timing: sdf %dms\n", (int)(t1_ms-t0_ms) );
		//#endif //DO_MULTIPLE_TIMINGS
	}
	
	#if defined ( DO_MULTIPLE_TIMINGS )
	std::sort( timings.begin(), timings.end() );
	uint64_t t_med = timings[ timings.size()/2 + 1 ];
	printf( "\n(min, med, max) = (%dms, %dms, %dms)\n\n", (int)t_min, (int)t_med, (int)t_max );
	#endif //DO_MULTIPLE_TIMINGS

	//#define DO_SANITY_CHECK
	#ifdef DO_SANITY_CHECK
	{
		PROFILE_SCOPE("sanity_check");
		printf( "\nsanity check...\n" );

		sdf_t sdf_bf;
		{
			init_sdf( &sdf_bf, bb, GRID_SIZ_X, GRID_SIZ_Y, GRID_SIZ_Z );
			const uint64_t t0_ms = gettime_ms();
			eval_sdf__bruteforce( sdf_bf, mesh );
			//eval_sdf__4grid_threaded( sdf_bf, mesh );
			//eval_sdf__precalc_simd_4grid( sdf_bf, mesh );
			const uint64_t t1_ms = gettime_ms();
			printf( "timing: sdf_bf %dms\n", (int)(t1_ms-t0_ms) );
		}

		bool sane = true;
		//{
		//	bool evalpoints_same = true;
		//	for ( size_t i=0,n=sdf.header.dim_x*sdf.header.dim_y*sdf.header.dim_z; i<n; ++i )
		//	{
		//		bool same = true;
		//		same = same && AlmostEqualRelative(sdf_bf.eval_points[i].x, sdf.eval_points[i].x );
		//		same = same && AlmostEqualRelative(sdf_bf.eval_points[i].y, sdf.eval_points[i].y );
		//		same = same && AlmostEqualRelative(sdf_bf.eval_points[i].z, sdf.eval_points[i].z );
		//		//if ( same == false )
		//		//	__debugbreak();
		//		evalpoints_same = evalpoints_same && same;
		//	}
		//	printf("evalpoints: %d\n", evalpoints_same );
		//	sane = sane && evalpoints_same;
		//}
		{
			assert( sdf_bf.header.dim_x == sdf.header.dim_x );
			assert( sdf_bf.header.dim_y == sdf.header.dim_y );
			assert( sdf_bf.header.dim_z == sdf.header.dim_z );

			float32_t mindiff =  FLT_MAX;
			float32_t maxdiff = -FLT_MAX;
			for ( int i=0,n=sdf_bf.header.dim_x*sdf_bf.header.dim_y*sdf_bf.header.dim_z; i<n; ++i )
			{
				const float32_t d0 = sdf_bf.data[ i ];
				const float32_t d1 = sdf.data[ i ];
				if ( !AlmostEqualRelative( d0, d1) )
				{
					float32_t diff = abs( d1 - d0 );
					if ( diff < mindiff ) mindiff = diff;
					if ( diff > maxdiff ) maxdiff = diff;
					sane = false;
				}
			}
			if( mindiff < maxdiff )
				printf( "delta: [min;max]=[%f;%f]", mindiff, maxdiff );
		}
		printf( "\nbf vs pc sane: %d", sane );
		
		printf("\n");

		_aligned_free( sdf_bf.data );
	}
	#endif //DO_SANITY_CHECK

	delete mesh;


	{
		FILE *f;
		fopen_s( &f, "sdf.bin", "wb" ); //TODO: write to <filename>_128x128x128_sdf.bin
		fwrite( &sdf.header, sizeof(header_t), 1, f );
		fwrite( sdf.data, sizeof(float32_t) * sdf.header.dim_x * sdf.header.dim_y * sdf.header.dim_z, 1, f );
		fclose( f );
	}


	//note: sanity check written data
	//{
	//	sdf_t sdf_sanity;
	//	FILE *f;
	//	fopen_s( &f, "sdf.bin", "rb" );
	//	fread( &sdf_sanity.header, sizeof(header_t), 1, f );
	//
	//	sdf_sanity.data = new float32_t[ sdf_sanity.header.dim_x * sdf_sanity.header.dim_y * sdf_sanity.header.dim_z ];
	//
	//	fread( sdf_sanity.data, sizeof(float32_t), sdf_sanity.header.dim_x * sdf_sanity.header.dim_y * sdf_sanity.header.dim_z, f );
	//	fclose(f);
	//
	//	//note: compare data
	//	for ( int i=0,n=sdf.header.dim_x*sdf.header.dim_y*sdf.header.dim_z; i<n; ++i )
	//	{
	//		const float32_t d = sdf.data[ i ];
	//		const float32_t d_sanity = sdf_sanity.data[ i ];
	//		if ( d != d_sanity )
	//		{
	//			printf( "ARGH1" );
	//		}
	//	}
	//
	//	for ( int z=0,zn=sdf.header.dim_z; z<zn; ++z ) {
	//	for ( int y=0,yn=sdf.header.dim_y; y<yn; ++y ) {
	//	for ( int x=0,xn=sdf.header.dim_x; x<xn; ++x )
	//	{
	//		const int idx = x + y*xn + z*xn*yn;
	//		const float32_t d = sdf.data[ idx ];
	//		const float32_t d_sanity = sdf_sanity.data[ idx ];
	//		if ( d != d_sanity )
	//		{
	//			printf( "ARGH2" );
	//		}
	//	}}}
	//
	//	delete [] sdf_sanity.data;
	//}	

	//const uint64_t t4_ms = gettime_ms();

	//printf( "timings: load %dms, sdf %dms, sdf_bf %dms\n", (int)(t11_ms-t0_ms), (int)(t12_ms-t11_ms), (int)(t22_ms-t21_ms) );

	_aligned_free( sdf.data );

	system("pause");

	return 0;
}

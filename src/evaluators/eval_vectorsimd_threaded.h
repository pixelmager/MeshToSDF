#pragma once

#include <sdf_support.h>
#include <evaluators/precalc.h>

// ====
struct workload_aos_parms_t
{
	__m128 p0;
	__m128 stepsiz;

	#ifndef NDEBUG
	int32_t threadidx;
	char const * threadname;
	#endif

	lpt::indexed_triangle_mesh_t const * mesh;
	tri_precalc_simd_aos_t const * tpc;
	sdf_t const * sdf;

	int32_t minidx;
	int32_t count;
};

// ====
void workload_aos( workload_aos_parms_t const * const parms )
{
	#ifndef NDEBUG
	PROFILE_THREADNAME( parms->threadname );
	#endif
	PROFILE_SCOPE("calc_idx");

	const int32_t dim_x = parms->sdf->header.dim_x;
	const int32_t dim_y = parms->sdf->header.dim_y;
	//const int32_t dim_z = parms->sdf->header.dim_z;

	for ( int z=parms->minidx, zn=parms->minidx+parms->count; z<zn; ++z )
	{
		for ( int y=0,yn=dim_y; y<yn; ++y )
		{
			for ( int x=0,xn=dim_x; x<xn; ++x )
			{
				__m128 xyz = _mm_set_ps( 0, (float)z, (float)y, (float)x ); //TODO: _mm_add_ps( p, _mm_shuffle_ps(stepsiz, stepsiz, 0300)) instead?
				__m128 mm_p = _mm_fmadd_ps( xyz, parms->stepsiz, parms->p0 );

				__m128 d_min = _mm_set1_ps( FLT_MAX );

				assert( parms->mesh->tri_indices.size() % 3 == 0 );
				for ( size_t idx_tri=0,num_tris=parms->mesh->tri_indices.size()/3; idx_tri<num_tris; ++idx_tri )
				{
					d_min = _mm_min_ps( d_min, udTriangle_sq_precalc_SIMD_aos( mm_p, parms->tpc[idx_tri] ) );
					//d_min = _mm_min_ps( d_min, udTriangle_sq_precalc( mm_p, parms->tpc[idx_tri] ) );
				}
				
				int idx = x + y*xn + z*xn*yn;
				parms->sdf->data[idx] = sqrtf( d_min.m128_f32[0] );
			}
		}
	}
}

// ====
void eval_sdf__aos_threaded( sdf_t &sdf, lpt::indexed_triangle_mesh_t const * const mesh )
{
	printf("%s\n", __FUNCTION__);

	aabb_t bb;
	bb.mn = vec3_t( sdf.header.bb_mn_x, sdf.header.bb_mn_y, sdf.header.bb_mn_z );
	bb.mx = vec3_t( sdf.header.bb_mx_x, sdf.header.bb_mx_y, sdf.header.bb_mx_z );
	vec3_t bb_range = bb.mx-bb.mn;

	tri_precalc_simd_aos_t * const tpc = precalc_simd_aos( mesh );

	const vec3_t stepsiz = vec3_t( bb_range.x / static_cast<float32_t>(sdf.header.dim_x),
								   bb_range.y / static_cast<float32_t>(sdf.header.dim_y),
								   bb_range.z / static_cast<float32_t>(sdf.header.dim_z) );
	
	const vec3_t p0 = bb.mn + 0.5f * stepsiz; //note: +0.5*stepsize to center at cell, -stepsiz for loop-init

	const int32_t num_cores = get_num_cores().num_cores_logical;
	const int num_threads = num_cores;

	std::vector<std::thread> threads;
	threads.reserve( num_threads );
	
	workload_aos_parms_t * const parms = new workload_aos_parms_t[ num_threads ];

	std::string *threadnames = new std::string[ num_threads ];

	int num_per_thread = static_cast<int>( ceilf( (float)sdf.header.dim_z / (float)num_threads) );
	for ( int i=0,n=num_threads; i<n; ++i )
	{
		//global
		{
			parms[i].mesh = mesh;
			parms[i].tpc = tpc;
			parms[i].sdf = &sdf;

			parms[i].p0 = _mm_set_ps(1, p0.z, p0.y, p0.x );
			parms[i].stepsiz = _mm_set_ps(0, stepsiz.z, stepsiz.y, stepsiz.x );
		}

		// per thread
		{
			parms[i].minidx = i * num_per_thread;
			parms[i].count  = min( num_per_thread, sdf.header.dim_z - parms[i].minidx );
		}

		//debug
		#ifndef NDEBUG
		{
			threadnames[i] = std::string("thread_") + lpt::to_string(i);
			//printf( "%s: [%d;%d[\n", threadnames[i].c_str(), parms[i].minidx, parms[i].minidx + parms[i].count );
			
			parms[i].threadidx = i; //threads[i].get_id();
			parms[i].threadname = threadnames[i].c_str();
		}
		#endif //NDEBUG

		if ( parms[i].count > 0 )
		{
			threads.push_back( std::thread( workload_aos, &(parms[i]) ) );
		}
	}

	for ( size_t i=0,n=threads.size(); i<n; ++i )
		threads[i].join();

	delete [] parms;
	delete [] threadnames;

	_aligned_free( tpc );
}


#pragma once

#include "eval_precalc.h"
#include <sdf_support.h>

// ====
struct workload_precalc_parms_t
{
	vec3_t p0;
	vec3_t stepsiz;

	#ifndef NDEBUG
	int32_t threadidx;
	char const * threadname;
	#endif

	lpt::indexed_triangle_mesh_t const * mesh;
	tri_precalc_t const * tpc;
	sdf_t const * sdf;

	int32_t minidx;
	int32_t count;
};

void workload_precalc( workload_precalc_parms_t const * const parms )
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
				const vec3_t xyz = vec3_t((float32_t)x, (float32_t)y, (float32_t)z);
				const vec3_t mm_p = xyz * parms->stepsiz + parms->p0;

				float d_min = FLT_MAX;

				assert( parms->mesh->tri_indices.size() % 3 == 0 );
				for ( size_t idx_tri=0,num_tris=parms->mesh->tri_indices.size()/3; idx_tri<num_tris; ++idx_tri )
				{
					d_min = lpt::min( d_min, udTriangle_sq_precalc( mm_p, parms->tpc[idx_tri] ) );
				}
				
				int idx = x + y*xn + z*xn*yn;
				parms->sdf->data[idx] = sqrtf( d_min );
			}
		}
	}
}
void eval_sdf__precalc_threaded( sdf_t &sdf, lpt::indexed_triangle_mesh_t const * const mesh )
{
	printf("%s\n", __FUNCTION__);

	aabb_t bb;
	bb.mn = vec3_t( sdf.header.bb_mn_x, sdf.header.bb_mn_y, sdf.header.bb_mn_z );
	bb.mx = vec3_t( sdf.header.bb_mx_x, sdf.header.bb_mx_y, sdf.header.bb_mx_z );
	vec3_t bb_range = bb.mx-bb.mn;

	tri_precalc_t * const tpc = precalc_tridata( mesh );

	const vec3_t stepsiz = vec3_t( bb_range.x / static_cast<float32_t>(sdf.header.dim_x),
								   bb_range.y / static_cast<float32_t>(sdf.header.dim_y),
								   bb_range.z / static_cast<float32_t>(sdf.header.dim_z) );
	
	const vec3_t p0 = bb.mn + 0.5f * stepsiz; //note: +0.5*stepsize to center at cell, -stepsiz for loop-init

	const int num_cores = get_num_cores();
	const int num_threads = num_cores;

	std::vector<std::thread> threads;
	threads.reserve( num_threads );
	
	workload_precalc_parms_t* const parms = new workload_precalc_parms_t[ num_threads ];

	std::string *threadnames = new std::string[ num_threads ];

	int num_per_thread = static_cast<int>( ceilf( (float)sdf.header.dim_z / (float)num_threads) );
	for ( int i=0,n=num_threads; i<n; ++i )
	{
		//global
		{
			parms[i].mesh = mesh;
			parms[i].tpc = tpc;
			parms[i].sdf = &sdf;

			parms[i].p0 = p0;
			parms[i].stepsiz = stepsiz;
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
			threads.push_back( std::thread( workload_precalc, &(parms[i]) ) );
		}
	}

	for ( size_t i=0,n=threads.size(); i<n; ++i )
		threads[i].join();

	delete [] parms;
	delete [] threadnames;

	_aligned_free( tpc );
}

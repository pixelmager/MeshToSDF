#pragma once

#include <sdf_support.h>
#include <evaluators/precalc.h>
#include <evaluators/eval_grid8.h>

struct workload_grid8_parms_t
{
	//note: constant across threads
	vec3_t p0;
	vec3_t stepsiz;
	lpt::indexed_triangle_mesh_t const * mesh;
	tri_precalc_t const * tpc;
	sdf_t const * sdf;

	//note: variable across threads
	int32_t minidx;
	int32_t count;
	int32_t threadidx;
	#ifndef NDEBUG
	char const * threadname;
	#endif //NDEBUG
};

// ====
void workload_grid8( workload_grid8_parms_t const * const parms )
{
    //note: use multiple windows threadgroups to get around 64-threads-per-group limit
	//      from https://twitter.com/id_aa_carmack/status/1249071471219150858?lang=en
	GROUP_AFFINITY group{};
	group.Mask = (KAFFINITY)-1;
	group.Group = parms->threadidx & 1;
	/*BOOL r =*/ SetThreadGroupAffinity(GetCurrentThread(), &group, nullptr);

	//#define VECTORIZE_GRIDCALC

	enum { SIMD_SIZ=8, SIMD_ALIGN=4*SIMD_SIZ };

	#ifndef NDEBUG
	PROFILE_THREADNAME( parms->threadname );
	#endif

	PROFILE_FUNC();

	float32_t *out_data = parms->sdf->data + parms->minidx;

	const int32_t dim_x = parms->sdf->header.dim_x;
	const int32_t dim_y = parms->sdf->header.dim_y;
	const int32_t dim_z = parms->sdf->header.dim_z;

	assert( parms->mesh->tri_indices.size() % 3 == 0 );
	const size_t num_tris=parms->mesh->tri_indices.size()/3;

	const __m256 p0_x = _mm256_set1_ps( parms->p0.x );
	const __m256 p0_y = _mm256_set1_ps( parms->p0.y );
	const __m256 p0_z = _mm256_set1_ps( parms->p0.z );
	const __m256 stepsiz_x = _mm256_set1_ps( parms->stepsiz.x );
	const __m256 stepsiz_y = _mm256_set1_ps( parms->stepsiz.y );
	const __m256 stepsiz_z = _mm256_set1_ps( parms->stepsiz.z );

	const __m256 D_MAX = _mm256_set1_ps( FLT_MAX );

	#if !defined(VECTORIZE_GRIDCALC)
	__m256i ii;
	for ( int i=0,n=SIMD_SIZ; i<n; ++i )
		ii.m256i_i32[i] = parms->minidx + i;
	#endif

	const __m256 fdx = _mm256_set1_ps( (float32_t)dim_x);
	const __m256 fdy = _mm256_set1_ps( (float32_t)dim_y);
	const __m256 fdz = _mm256_set1_ps( (float32_t)dim_z);
	const __m256 fdxy = _mm256_mul_ps(fdx, fdy);
	const __m256 rcp_fdx = _mm256_rcp_ps(fdx);
	const __m256 rcp_fdxy = _mm256_rcp_ps(fdxy);
		
	#if defined(VECTORIZE_GRIDCALC)
	const __m256 rcp_fdx_half = _mm256_mul_ps( rcp_fdx, _mm256_set1_ps(0.5f) ); //TODO: HACK float-precision
	const __m256 rcp_fdxy_half = _mm256_mul_ps( rcp_fdxy, _mm256_set1_ps(0.5f) );  //TODO: HACK float-precision

	__m256 fi;
	for ( int i=0,n=SIMD_SIZ; i<n; ++i )
		fi.m256_f32[i] = (float32_t)(parms->minidx + i);
	#endif

	assert( parms->minidx % SIMD_SIZ == 0 );
	assert( parms->count % SIMD_SIZ == 0 );
	const int32_t block_startidx = parms->minidx / SIMD_SIZ;
	const int32_t block_count = parms->count / SIMD_SIZ;
	for ( int32_t idx_block=block_startidx, nidx_block=block_startidx+block_count; idx_block<nidx_block; ++idx_block )
	{
		__m256 idx_x, idx_y, idx_z;

		#if !defined(VECTORIZE_GRIDCALC)
		//__m256 idx0_x, idx0_y, idx0_z;
		//TODO: if dim_* are 2^n, the grid-indices calculation is just bit-mask+offset
		for ( int i=0,n=SIMD_SIZ; i<n; ++i )
		{
			//int ii = SIMD_SIZ*idx_block + i;
			int32_t ix =  ii.m256i_i32[i] % (dim_x);
			int32_t iy = (ii.m256i_i32[i] / dim_x) % dim_y;
			int32_t iz =  ii.m256i_i32[i] / (dim_x*dim_y);
		
			idx_x.m256_f32[i] = (float32_t)ix;
			idx_y.m256_f32[i] = (float32_t)iy;
			idx_z.m256_f32[i] = (float32_t)iz;
		}
		#endif

		#if defined(VECTORIZE_GRIDCALC)
		{
			//note: example
			//2x2x2=8
			//idx=5 => (x,y,z)=(0,1,1)
			//iz = idx/(x*y)           = floor( 5/(2*2) ) = 1
			//iy = (idx - iz*x*y)/x    = floor( (5-1*2*2)/2 )  = floor( (5-4)/2 ) = 0
			//ix = idx - iy*x - iz*x*y = floor( 5 - 0*2 - 1*2*2 ) = 5-4 = 1
			
			//iz = idx/(x*y)
			idx_z = _mm256_fmadd_ps( fi, rcp_fdxy, rcp_fdxy_half ); //TODO: HACK, float-precision
			//idx_z = _mm256_cvtepi32_ps(_mm256_cvtps_epi32(idx_z)); //note: floor
            idx_z = _mm256_round_ps(idx_z, _MM_FROUND_FLOOR);
		
			//iy = (idx - iz*x*y)/x
			__m256 izxy = _mm256_mul_ps(idx_z, fdxy);
			idx_y = _mm256_sub_ps(fi, izxy);
			idx_y = _mm256_fmadd_ps( idx_y, rcp_fdx, rcp_fdx_half ); //TODO: HACK, float-precision
			//idx_y = _mm256_cvtepi32_ps(_mm256_cvtps_epi32(idx_y)); //note: floor
            idx_y = _mm256_round_ps(idx_y, _MM_FROUND_FLOOR);
		
			//ix = idx - iy*y - iz*x*y
			__m256 izxy_iyx = _mm256_fmadd_ps(idx_y, fdx, izxy);
			idx_x = _mm256_sub_ps( fi, izxy_iyx );
			//idx_x = _mm256_cvtepi32_ps(_mm256_cvtps_epi32(idx_x)); //note: floor
		}
		#endif

		//note: debug-check
		//for ( int i=0,n=SIMD_SIZ; i<n; ++i )
		//{
		//	assert( idx0_x.m256_f32[i] == idx_x.m256_f32[i] );
		//	assert( idx0_y.m256_f32[i] == idx_y.m256_f32[i] );
		//	assert( idx0_z.m256_f32[i] == idx_z.m256_f32[i] );
		//}

		const __m256 p_x = _mm256_fmadd_ps( stepsiz_x, idx_x, p0_x );
		const __m256 p_y = _mm256_fmadd_ps( stepsiz_y, idx_y, p0_y );
		const __m256 p_z = _mm256_fmadd_ps( stepsiz_z, idx_z, p0_z );

		//#ifndef NDEBUG
		//parms->sdf->eval_points[SIMD_SIZ*idx_block + i] = vec3_t( p_x.m256_f32[i], p_y.m256_f32[i], p_z.m256_f32[i] );
		//#endif //NDEBUG

		__m256 d_min = D_MAX;
		for ( size_t idx_tri=0; idx_tri<num_tris; ++idx_tri )
		{
			d_min = _mm256_min_ps( d_min, udTriangle_sq_precalc_SIMD_8grid( p_x, p_y, p_z, parms->tpc[idx_tri] ) ); //aos
		}
		d_min = _mm256_sqrt_ps(d_min);

		//_mm256_store_ps( out_data, d_min );
		_mm256_stream_ps( out_data, d_min );

		out_data += SIMD_SIZ;

		#if !defined(VECTORIZE_GRIDCALC)
		ii = _mm256_add_epi32( ii, _mm256_set1_epi32(SIMD_SIZ) );
		#endif

		#if defined(VECTORIZE_GRIDCALC)
		fi = _mm256_add_ps( fi, _mm256_set1_ps(SIMD_SIZ) );
		#endif
	}
}

// ====
void eval_sdf__grid8_threaded( sdf_t &sdf, lpt::indexed_triangle_mesh_t const * const mesh, int32_t num_threads )
{
	PROFILE_FUNC();

	//printf("%s\n", __FUNCTION__);

	enum { SIMD_SIZ=8, SIMD_ALIGN=4*SIMD_SIZ, BLOCK_SIZ=SIMD_SIZ };

	PROFILE_ENTER("precalc");

	aabb_t bb;
	bb.mn = vec3_t( sdf.header.bb_mn_x, sdf.header.bb_mn_y, sdf.header.bb_mn_z );
	bb.mx = vec3_t( sdf.header.bb_mx_x, sdf.header.bb_mx_y, sdf.header.bb_mx_z );
	vec3_t bb_range = bb.mx-bb.mn;

	const vec3_t stepsiz = vec3_t( bb_range.x / static_cast<float32_t>(sdf.header.dim_x),
								   bb_range.y / static_cast<float32_t>(sdf.header.dim_y),
								   bb_range.z / static_cast<float32_t>(sdf.header.dim_z) );
	
	const vec3_t p0 = bb.mn + 0.5f * stepsiz; //note: +0.5*stepsize to center at cell

	std::vector<std::thread> threads;
	threads.reserve( num_threads );

	tri_precalc_t * const tpc = precalc_tridata( mesh );

	PROFILE_LEAVE("precalc");
	PROFILE_ENTER("spawn threads");

	workload_grid8_parms_t * const parms = (workload_grid8_parms_t*)_aligned_malloc( num_threads*sizeof(workload_grid8_parms_t), SIMD_ALIGN );

	#ifndef NDEBUG
	std::string *threadnames = new std::string[ num_threads ];
	#endif //NDEBUG
	
	const int32_t num_cells             = sdf.header.dim_x * sdf.header.dim_y * sdf.header.dim_z;
	const int32_t num_blocks            = num_cells / BLOCK_SIZ;
	const int32_t num_blocks_per_thread = static_cast<int>( ceilf( (float)num_blocks/ (float)num_threads) );
	const int32_t num_cells_per_thread  = num_blocks_per_thread * BLOCK_SIZ;
	const int32_t num_cells_evaluated   = num_blocks * BLOCK_SIZ;
	const int32_t num_cells_remaining   = num_cells - num_cells_evaluated;
	for ( int idx_thread=0,n_thread=num_threads; idx_thread<n_thread; ++idx_thread )
	{
		// per thread
		{
            assert( idx_thread < num_threads );
			parms[idx_thread].minidx = idx_thread * num_cells_per_thread;
			parms[idx_thread].count  = min( num_cells_per_thread, num_cells_evaluated - parms[idx_thread].minidx );
			assert( parms[idx_thread].count % SIMD_SIZ == 0 );
			if ( parms[idx_thread].count < 1 )
				continue;
		}

		//global
		//TODO: move to separate struct
		{
			parms[idx_thread].threadidx = idx_thread; //threads[i].get_id();

			parms[idx_thread].p0 = p0;
			parms[idx_thread].stepsiz = stepsiz;

			parms[idx_thread].mesh = mesh;
			parms[idx_thread].tpc = tpc;
			parms[idx_thread].sdf = &sdf; //TODO: test separately allocated outputs (to not get cache-overwrites)
		}

		//debug
		#ifndef NDEBUG
		{
			threadnames[idx_thread] = std::string("thread_") + lpt::to_string(idx_thread);
			printf( "\"%s\" [%d;%d[ count=%d\n", threadnames[idx_thread].c_str(), parms[idx_thread].minidx, parms[idx_thread].minidx + parms[idx_thread].count, parms[idx_thread].count );
			parms[idx_thread].threadname = threadnames[idx_thread].c_str();
		}
		#endif //NDEBUG

		threads.push_back( std::thread( workload_grid8, &(parms[idx_thread]) ) );
	}
	#ifndef NDEBUG
	printf( "# spawned %d/%d threads\n  remaining cells: %d\n", (int)threads.size(), num_threads, num_cells_remaining );
	#endif

	PROFILE_LEAVE("spawn threads");

	//note: process num_cells_remaining on this thread
	if ( num_cells_remaining > 0 )
	{
		PROFILE_SCOPE("remaining cells");

		assert( num_cells_remaining < SIMD_SIZ );
		__m256 p_x = _mm256_set1_ps(0.0f);
		__m256 p_y = _mm256_set1_ps(0.0f);
		__m256 p_z = _mm256_set1_ps(0.0f);

		const int32_t startidx = sdf.header.dim_x * sdf.header.dim_y * sdf.header.dim_z - num_cells_remaining;
		for ( int i=0,n=num_cells_remaining; i<n; ++i )
		{
			int32_t ii = startidx + i;
			const int32_t ix =  ii % sdf.header.dim_x;
			const int32_t iy = (ii / sdf.header.dim_x) % sdf.header.dim_y;
			const int32_t iz =  ii / (sdf.header.dim_x * sdf.header.dim_y);

			p_x.m256_f32[i] = p0.x + stepsiz.x * (float32_t)ix;
			p_y.m256_f32[i] = p0.y + stepsiz.y * (float32_t)iy;
			p_z.m256_f32[i] = p0.z + stepsiz.z * (float32_t)iz;
		}
		__m256 d_min = _mm256_set1_ps( FLT_MAX );
		for ( size_t idx_tri=0; idx_tri<mesh->tri_indices.size()/3; ++idx_tri )
		{
			d_min = _mm256_min_ps( d_min, udTriangle_sq_precalc_SIMD_8grid( p_x, p_y, p_z, parms->tpc[idx_tri] ) );
		}
		d_min = _mm256_sqrt_ps(d_min);
		for ( int i=0,n=num_cells_remaining; i<n; ++i )
		{
			int32_t ii = startidx + i;
			sdf.data[ ii ] = d_min.m256_f32[ i ];
		}
	}

 	for ( size_t i=0,n=threads.size(); i<n; ++i )
	{
		threads[i].join();
	}
	threads.clear();

	_aligned_free( parms );

	#ifndef NDEBUG
	delete [] threadnames;
	#endif //NDEBUG

	_aligned_free( tpc );
}

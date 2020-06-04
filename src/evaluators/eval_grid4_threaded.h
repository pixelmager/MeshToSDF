#pragma once

#include <sdf_support.h>
#include <evaluators/precalc.h>

struct workload_grid4_parms_t
{
	//note: constant across threads
	vec3_t p0;
	vec3_t stepsiz;
	lpt::indexed_triangle_mesh_t const * mesh;
	tri_precalc_interleaved_t const * tpc;
	sdf_t const * sdf;

	//note: variable across threads
	int32_t minidx;
	int32_t count;
	#ifndef NDEBUG
	int32_t threadidx;
	char const * threadname;
	#endif //NDEBUG
};

// ====
void workload_grid4( workload_grid4_parms_t const * const parms )
{
	enum { SIMD_SIZ=4, SIMD_ALIGN=4*SIMD_SIZ };

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

	const __m128 p0_x = _mm_set1_ps( parms->p0.x );
	const __m128 p0_y = _mm_set1_ps( parms->p0.y );
	const __m128 p0_z = _mm_set1_ps( parms->p0.z );
	const __m128 stepsiz_x = _mm_set1_ps( parms->stepsiz.x );
	const __m128 stepsiz_y = _mm_set1_ps( parms->stepsiz.y );
	const __m128 stepsiz_z = _mm_set1_ps( parms->stepsiz.z );

	const __m128 D_MAX = _mm_set1_ps( FLT_MAX );

	__m128i ofs;
	for ( int i=0,n=SIMD_SIZ; i<n; ++i )
		ofs.m128i_i32[i] = parms->minidx + i;
	__m128i ii = ofs;

	assert( parms->minidx % SIMD_SIZ == 0 );
	assert( parms->count % SIMD_SIZ == 0 );
	const int32_t block_startidx = parms->minidx / SIMD_SIZ;
	const int32_t block_count = parms->count / SIMD_SIZ;
	for ( int32_t idx_block=block_startidx, nidx_block=block_startidx+block_count; idx_block<nidx_block; ++idx_block )
	{
		__m128 idx_x;
		__m128 idx_y;
		__m128 idx_z;

		//TODO: vectorize
		for ( int i=0,n=SIMD_SIZ; i<n; ++i )
		{
			//int ii = SIMD_SIZ*idx_block + i;
			int32_t ix =  ii.m128i_i32[i] % (dim_x);
			int32_t iy = (ii.m128i_i32[i] / dim_x) % dim_y;
			int32_t iz =  ii.m128i_i32[i] / (dim_x*dim_y);
		
			idx_x.m128_f32[i] = (float32_t)ix;
			idx_y.m128_f32[i] = (float32_t)iy;
			idx_z.m128_f32[i] = (float32_t)iz;
		}
		//__m128 ii = _mm_fmadd_ps( 

		const __m128 p_x = _mm_fmadd_ps( stepsiz_x, idx_x, p0_x );
		const __m128 p_y = _mm_fmadd_ps( stepsiz_y, idx_y, p0_y );
		const __m128 p_z = _mm_fmadd_ps( stepsiz_z, idx_z, p0_z );

		//#ifndef NDEBUG
		//parms->sdf->eval_points[SIMD_SIZ*idx_block + i] = vec3_t( p_x.m128_f32[i], p_y.m128_f32[i], p_z.m128_f32[i] );
		//#endif //NDEBUG

		__m128 d_min = D_MAX;
		for ( size_t idx_tri=0; idx_tri<num_tris; ++idx_tri )
		{
			d_min = _mm_min_ps( d_min, udTriangle_sq_precalc_SIMD_4grid( p_x, p_y, p_z, parms->tpc[idx_tri] ) ); //aos
		}
		d_min = _mm_sqrt_ps(d_min);

		_mm_store_ps( out_data, d_min );
		out_data += SIMD_SIZ;

		ii = _mm_add_epi32( ii, _mm_set1_epi32(SIMD_SIZ) );
	}
}

// ====
void eval_sdf__grid4_threaded( sdf_t &sdf, lpt::indexed_triangle_mesh_t const * const mesh )
{
    printf("%s (sse)\n", __FUNCTION__);

	PROFILE_FUNC();

	enum { SIMD_SIZ=4, SIMD_ALIGN=4*SIMD_SIZ, BLOCK_SIZ=SIMD_SIZ };

	aabb_t bb;
	bb.mn = vec3_t( sdf.header.bb_mn_x, sdf.header.bb_mn_y, sdf.header.bb_mn_z );
	bb.mx = vec3_t( sdf.header.bb_mx_x, sdf.header.bb_mx_y, sdf.header.bb_mx_z );
	vec3_t bb_range = bb.mx-bb.mn;

	const vec3_t stepsiz = vec3_t( bb_range.x / static_cast<float32_t>(sdf.header.dim_x),
								   bb_range.y / static_cast<float32_t>(sdf.header.dim_y),
								   bb_range.z / static_cast<float32_t>(sdf.header.dim_z) );
	
	const vec3_t p0 = bb.mn + 0.5f * stepsiz; //note: +0.5*stepsize to center at cell

	const int32_t num_cores = get_num_cores().num_cores_logical;
	int32_t num_hwthreads = num_cores;

	std::vector<std::thread> threads;
	threads.reserve( num_hwthreads );

	//tri_precalc_simd_soa_t * const tpc = precalc_simd_soa( mesh );
	//tri_precalc_simd_aos_t * const tpc = precalc_simd_aos( mesh );
	tri_precalc_interleaved_t * const tpc = precalc_tridata_interleaved( mesh );

	PROFILE_ENTER("spawn threads");
	workload_grid4_parms_t * const parms = (workload_grid4_parms_t*)_aligned_malloc( num_hwthreads*sizeof(workload_grid4_parms_t), SIMD_ALIGN );

	#ifndef NDEBUG
	std::string *threadnames = new std::string[ num_hwthreads ];
	#endif //NDEBUG
	
	const int32_t num_cells             = sdf.header.dim_x * sdf.header.dim_y * sdf.header.dim_z;
	const int32_t num_blocks            = num_cells / BLOCK_SIZ;
	const int32_t num_blocks_per_thread = static_cast<int>( ceilf( (float)num_blocks/ (float)num_hwthreads) );
	const int32_t num_cells_per_thread  = num_blocks_per_thread * BLOCK_SIZ;
	const int32_t num_cells_evaluated   = num_blocks * BLOCK_SIZ;
	const int32_t num_cells_remaining   = num_cells - num_cells_evaluated;
	for ( int idx_thread=0,n_thread=num_hwthreads; idx_thread<n_thread; ++idx_thread )
	{
		// per thread
		{
            assert( idx_thread < num_hwthreads );
			parms[idx_thread].minidx = idx_thread * num_cells_per_thread;
			parms[idx_thread].count  = min( num_cells_per_thread, num_cells_evaluated - parms[idx_thread].minidx );
			assert( parms[idx_thread].count % SIMD_SIZ == 0 );
			if ( parms[idx_thread].count < 1 )
				continue;
		}

		//global
		//TODO: move to separate struct
		{
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

			parms[idx_thread].threadidx = idx_thread; //threads[i].get_id();
			parms[idx_thread].threadname = threadnames[idx_thread].c_str();
		}
		#endif //NDEBUG

		threads.push_back( std::thread( workload_grid4, &(parms[idx_thread]) ) );
	}
	#ifndef NDEBUG
	printf( "# spawned %d/%d threads\n  remaining cells: %d\n", (int)threads.size(), num_hwthreads, num_cells_remaining );
	#endif

	PROFILE_LEAVE("spawn threads");
	PROFILE_ENTER("remaining cells");

	//note: process num_cells_remaining on this thread
	if ( num_cells_remaining > 0 )
	{
		
		assert( num_cells_remaining < SIMD_SIZ );
		__m128 p_x = _mm_set1_ps(0.0f);
		__m128 p_y = _mm_set1_ps(0.0f);
		__m128 p_z = _mm_set1_ps(0.0f);

		const int32_t startidx = sdf.header.dim_x * sdf.header.dim_y * sdf.header.dim_z - num_cells_remaining;
		for ( int i=0,n=num_cells_remaining; i<n; ++i )
		{
			int32_t ii = startidx + i;
			const int32_t ix =  ii % sdf.header.dim_x;
			const int32_t iy = (ii / sdf.header.dim_x) % sdf.header.dim_y;
			const int32_t iz =  ii / (sdf.header.dim_x * sdf.header.dim_y);

			p_x.m128_f32[i] = p0.x + stepsiz.x * (float32_t)ix;
			p_y.m128_f32[i] = p0.y + stepsiz.y * (float32_t)iy;
			p_z.m128_f32[i] = p0.z + stepsiz.z * (float32_t)iz;
		}
		__m128 d_min = _mm_set1_ps( FLT_MAX );
		for ( size_t idx_tri=0; idx_tri<mesh->tri_indices.size()/3; ++idx_tri )
		{
			//d_min = _mm_min_ps( d_min, udTriangle_sq_precalc_SIMD_4grid( p_x, p_y, p_z, parms->tpc, idx_tri ) );
			d_min = _mm_min_ps( d_min, udTriangle_sq_precalc_SIMD_4grid( p_x, p_y, p_z, parms->tpc[idx_tri] ) );
		}
		d_min = _mm_sqrt_ps(d_min);
		for ( int i=0,n=num_cells_remaining; i<n; ++i )
		{
			int32_t ii = startidx + i;
			sdf.data[ ii ] = d_min.m128_f32[ i ];
		}
	}
	PROFILE_LEAVE("remaining cells");

 	for ( size_t i=0,n=threads.size(); i<n; ++i )
	{
		threads[i].join();
	}

	_aligned_free( parms );

	#ifndef NDEBUG
	delete [] threadnames;
	#endif //NDEBUG

	_aligned_free( tpc );
}


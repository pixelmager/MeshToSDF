#pragma once

#include <sdf_support.h>

using namespace lpt;

struct tri_precalc_t
{
	vec3_t v1;
	vec3_t v2;
	vec3_t v3;
	vec3_t v21;
	vec3_t v32;
	vec3_t v13;

	vec3_t nor;
	vec3_t cp0;
	vec3_t cp1;
	vec3_t cp2;

	vec3_t rcp_dp2_v21_v32_v13;

	float32_t rcp_dp2_nor;
};

tri_precalc_t* precalc_tridata( lpt::indexed_triangle_mesh_t const * const mesh )
{
	PROFILE_FUNC();
	
	tri_precalc_t * const tpc = (tri_precalc_t*)_aligned_malloc( sizeof(tri_precalc_t) * (mesh->tri_indices.size() / 3 ), 16 );

	assert( mesh->positions.size() % 3 == 0 );
	vec3_t const * const positions = reinterpret_cast<vec3_t const * const>( &mesh->positions[0] );

	for ( size_t idx_tri=0,num_tris=mesh->tri_indices.size()/3; idx_tri<num_tris; ++idx_tri )
	{
		const uint32_t idx0 = mesh->tri_indices[ 3*idx_tri+0 ];
		const uint32_t idx1 = mesh->tri_indices[ 3*idx_tri+1 ];
		const uint32_t idx2 = mesh->tri_indices[ 3*idx_tri+2 ];
		const vec3_t &pos_v1 = positions[ idx0 ];
		const vec3_t &pos_v2 = positions[ idx1 ];
		const vec3_t &pos_v3 = positions[ idx2 ];

		tri_precalc_t &pc = tpc[ idx_tri ];
		pc.v1 = pos_v1;
		pc.v2 = pos_v2;
		pc.v3 = pos_v3;

		pc.v21 = sub(pc.v2, pc.v1);
		pc.v32 = sub(pc.v3, pc.v2);
		pc.v13 = sub(pc.v1, pc.v3);

		pc.nor = cross(pc.v21, pc.v13);

		pc.cp0 = cross(pc.v21, pc.nor);
		pc.cp1 = cross(pc.v32, pc.nor);
		pc.cp2 = cross(pc.v13, pc.nor);

		pc.rcp_dp2_v21_v32_v13 = vec3_t(1.0f / length_sq(pc.v21),
										1.0f / length_sq(pc.v32),
										1.0f / length_sq(pc.v13) );

		pc.rcp_dp2_nor = 1.0f / length_sq(pc.nor);
	}

	return tpc;
}

///////////////////////////////////////////////////////////////////////////////

struct tri_precalc_interleaved_t
{
	vec3_t v1;
	vec3_t v2;
	vec3_t v3;

	vec3_t nor;

	vec3_t cp0;
	vec3_t cp1;
	vec3_t cp2;

	vec3_t rcp_dp2_v21_v32_v13;

	float32_t rcp_dp2_nor;
};
tri_precalc_interleaved_t* precalc_tridata_interleaved( lpt::indexed_triangle_mesh_t const * const mesh )
{
	PROFILE_FUNC();

	enum { SIMD_SIZ=16, SIMD_ALIGN=4*SIMD_SIZ };
	
	assert( mesh->tri_indices.size() % 3 == 0 );
	const size_t num_tris = mesh->tri_indices.size() / 3;
	tri_precalc_interleaved_t * const tpc = (tri_precalc_interleaved_t*)_aligned_malloc( sizeof(tri_precalc_interleaved_t) * num_tris, SIMD_ALIGN );

	assert( mesh->positions.size() % 3 == 0 );
	vec3_t const * const positions = reinterpret_cast<vec3_t const * const>( &mesh->positions[0] );

	for ( size_t idx_tri=0; idx_tri<num_tris; ++idx_tri )
	{
		const uint32_t idx0 = mesh->tri_indices[ 3*idx_tri+0 ];
		const uint32_t idx1 = mesh->tri_indices[ 3*idx_tri+1 ];
		const uint32_t idx2 = mesh->tri_indices[ 3*idx_tri+2 ];
		const vec3_t &pos_v1 = positions[ idx0 ];
		const vec3_t &pos_v2 = positions[ idx1 ];
		const vec3_t &pos_v3 = positions[ idx2 ];

		tri_precalc_interleaved_t &pc = tpc[ idx_tri ];
		pc.v1 = pos_v1;
		pc.v2 = pos_v2;
		pc.v3 = pos_v3;

		const vec3_t v21 = sub(pc.v2, pc.v1);
		const vec3_t v32 = sub(pc.v3, pc.v2);
		const vec3_t v13 = sub(pc.v1, pc.v3);

		pc.nor = cross(v21, v13);

		pc.cp0 = cross(v21, pc.nor);
		pc.cp1 = cross(v32, pc.nor);
		pc.cp2 = cross(v13, pc.nor);

		pc.rcp_dp2_v21_v32_v13 = vec3_t(1.0f / length_sq(v21),
										1.0f / length_sq(v32),
										1.0f / length_sq(v13) );

		pc.rcp_dp2_nor = 1.0f / length_sq(pc.nor);
	}

	return tpc;
}

///////////////////////////////////////////////////////////////////////////////

struct tri_precalc_simd_aos_t
{
	ALIGN16 __m128 v1;
	ALIGN16 __m128 v2;
	ALIGN16 __m128 v3;

	ALIGN16 __m128 v21;
	ALIGN16 __m128 v32;
	ALIGN16 __m128 v13;

	ALIGN16 __m128 rcp_dp2_v21_v32_v13;

	ALIGN16 __m128 nor;

	ALIGN16 __m128 rcp_dp2_nor; //xxxx

	ALIGN16 __m128 cp0;
	ALIGN16 __m128 cp1;
	ALIGN16 __m128 cp2;
};

tri_precalc_simd_aos_t* precalc_simd_aos( lpt::indexed_triangle_mesh_t const * const mesh )
{
	PROFILE_FUNC();
	
	tri_precalc_simd_aos_t * const tpc = (tri_precalc_simd_aos_t*)_aligned_malloc( sizeof(tri_precalc_simd_aos_t) * (mesh->tri_indices.size() / 3 ), 16 );

	assert( mesh->positions.size() % 3 == 0 );
	vec3_t const * const positions = reinterpret_cast<vec3_t const * const>( &mesh->positions[0] );

	for ( size_t idx_tri=0,num_tris=mesh->tri_indices.size()/3; idx_tri<num_tris; ++idx_tri )
	{
		const uint32_t idx0 = mesh->tri_indices[ 3*idx_tri+0 ];
		const uint32_t idx1 = mesh->tri_indices[ 3*idx_tri+1 ];
		const uint32_t idx2 = mesh->tri_indices[ 3*idx_tri+2 ];
		const vec3_t &pos_v1 = positions[ idx0 ];
		const vec3_t &pos_v2 = positions[ idx1 ];
		const vec3_t &pos_v3 = positions[ idx2 ];

		tri_precalc_simd_aos_t &pc = tpc[ idx_tri ];
		vec4_t v1(pos_v1, 1);
		vec4_t v2(pos_v2, 1);
		vec4_t v3(pos_v3, 1);
		pc.v1 = _mm_load_ps( v1.ptr() );
		pc.v2 = _mm_load_ps( v2.ptr() );
		pc.v3 = _mm_load_ps( v3.ptr() );

		vec4_t v21 = sub(v2, v1);
		vec4_t v32 = sub(v3, v2);
		vec4_t v13 = sub(v1, v3);
		pc.v21 = _mm_load_ps( v21.ptr() );
		pc.v32 = _mm_load_ps( v32.ptr() );
		pc.v13 = _mm_load_ps( v13.ptr() );

		vec4_t nor( cross(v21.xyz(), v13.xyz()), 0 );
		pc.nor = _mm_load_ps( nor.ptr() );

		vec3_t cp0 = cross(v21.xyz(), nor.xyz());
		vec3_t cp1 = cross(v32.xyz(), nor.xyz());
		vec3_t cp2 = cross(v13.xyz(), nor.xyz());
		pc.cp0 = _mm_load_ps( vec4_t( cp0, 0 ).ptr() );
		pc.cp1 = _mm_load_ps( vec4_t( cp1, 0 ).ptr() );
		pc.cp2 = _mm_load_ps( vec4_t( cp2, 0 ).ptr() );

		pc.rcp_dp2_v21_v32_v13 = _mm_load_ps( vec4_t(1.0f / length_sq(v21),
													 1.0f / length_sq(v32),
													 1.0f / length_sq(v13),
													 0).ptr() );

		pc.rcp_dp2_nor = _mm_set1_ps( 1.0f / length_sq(nor) );
	}

	return tpc;
}

///////////////////////////////////////////////////////////////////////////////

struct tri_precalc_simd_soa_t
{
	float32_t *v1_x;
	float32_t *v1_y;
	float32_t *v1_z;

	float32_t *v2_x;
	float32_t *v2_y;
	float32_t *v2_z;

	float32_t *v3_x;
	float32_t *v3_y;
	float32_t *v3_z;

	//float32_t *v21_x;
	//float32_t *v21_y;
	//float32_t *v21_z;

	float32_t *v32_x;
	float32_t *v32_y;
	float32_t *v32_z;

	float32_t *v13_x;
	float32_t *v13_y;
	float32_t *v13_z;

	float32_t *rcp_dp2_v21;
	float32_t *rcp_dp2_v32;
	float32_t *rcp_dp2_v13;

	float32_t *nor_x;
	float32_t *nor_y;
	float32_t *nor_z;

	float32_t *rcp_dp2_nor;

	float32_t *cp0_x;
	float32_t *cp0_y;
	float32_t *cp0_z;

	float32_t *cp1_x;
	float32_t *cp1_y;
	float32_t *cp1_z;

	float32_t *cp2_x;
	float32_t *cp2_y;
	float32_t *cp2_z;
};

tri_precalc_simd_soa_t* precalc_simd_soa( lpt::indexed_triangle_mesh_t const * const mesh )
{
	enum { SIMD_SIZ=16, SIMD_ALIGN=4*SIMD_SIZ };
	PROFILE_FUNC();
	
	tri_precalc_simd_soa_t * const tpc = (tri_precalc_simd_soa_t*)_aligned_malloc( sizeof(tri_precalc_simd_soa_t), SIMD_ALIGN );

	assert( mesh->positions.size() % 3 == 0 );
	const size_t num_tris = mesh->tri_indices.size() / 3;

	//TODO: could potentially be allocated as one big block, but might be unaligned for each array depending on number of triangles?
	tpc->v1_x = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	tpc->v1_y = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	tpc->v1_z = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );

	tpc->v2_x = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	tpc->v2_y = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	tpc->v2_z = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );

	tpc->v3_x = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	tpc->v3_y = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	tpc->v3_z = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );

	//tpc->v21_x = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	//tpc->v21_y = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	//tpc->v21_z = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );

	//tpc->v32_x = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	//tpc->v32_y = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	//tpc->v32_z = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );

	//tpc->v13_x = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	//tpc->v13_y = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	//tpc->v13_z = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );

	tpc->rcp_dp2_v21 = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	tpc->rcp_dp2_v32 = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	tpc->rcp_dp2_v13 = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );

	tpc->nor_x = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	tpc->nor_y = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	tpc->nor_z = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );

	tpc->rcp_dp2_nor = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );

	tpc->cp0_x = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	tpc->cp0_y = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	tpc->cp0_z = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );

	tpc->cp1_x = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	tpc->cp1_y = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	tpc->cp1_z = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );

	tpc->cp2_x = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	tpc->cp2_y = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );
	tpc->cp2_z = (float32_t*)_aligned_malloc( sizeof(float32_t) * num_tris, SIMD_ALIGN );

	vec3_t const * const positions = reinterpret_cast<vec3_t const * const>( &mesh->positions[0] );

	//TODO: vectorize
	for ( size_t idx_tri=0; idx_tri<num_tris; ++idx_tri )
	{
		const uint32_t idx0 = mesh->tri_indices[ 3*idx_tri+0 ];
		const uint32_t idx1 = mesh->tri_indices[ 3*idx_tri+1 ];
		const uint32_t idx2 = mesh->tri_indices[ 3*idx_tri+2 ];
		assert( 3*idx0 < mesh->positions.size() );
		assert( 3*idx1 < mesh->positions.size() );
		assert( 3*idx2 < mesh->positions.size() );

		const vec3_t &pos_v1 = positions[ idx0 ];
		const vec3_t &pos_v2 = positions[ idx1 ];
		const vec3_t &pos_v3 = positions[ idx2 ];

		vec4_t v1(pos_v1, 1);
		vec4_t v2(pos_v2, 1);
		vec4_t v3(pos_v3, 1);
		tpc->v1_x[idx_tri] = v1.x;
		tpc->v1_y[idx_tri] = v1.y;
		tpc->v1_z[idx_tri] = v1.z;
		tpc->v2_x[idx_tri] = v2.x;
		tpc->v2_y[idx_tri] = v2.y;
		tpc->v2_z[idx_tri] = v2.z;
		tpc->v3_x[idx_tri] = v3.x;
		tpc->v3_y[idx_tri] = v3.y;
		tpc->v3_z[idx_tri] = v3.z;

		vec4_t v21 = sub(v2, v1);
		vec4_t v32 = sub(v3, v2);
		vec4_t v13 = sub(v1, v3);
		//tpc->v21_x[idx_tri] = v21.x;
		//tpc->v21_y[idx_tri] = v21.y;
		//tpc->v21_z[idx_tri] = v21.z;
		//tpc->v32_x[idx_tri] = v32.x;
		//tpc->v32_y[idx_tri] = v32.y;
		//tpc->v32_z[idx_tri] = v32.z;
		//tpc->v13_x[idx_tri] = v13.x;
		//tpc->v13_y[idx_tri] = v13.y;
		//tpc->v13_z[idx_tri] = v13.z;

		vec4_t nor( cross(v21.xyz(), v13.xyz()), 0 );
		tpc->nor_x[idx_tri] = nor.x;
		tpc->nor_y[idx_tri] = nor.y;
		tpc->nor_z[idx_tri] = nor.z;

		vec3_t cp0 = cross(v21.xyz(), nor.xyz());
		vec3_t cp1 = cross(v32.xyz(), nor.xyz());
		vec3_t cp2 = cross(v13.xyz(), nor.xyz());
		tpc->cp0_x[idx_tri] = cp0.x;
		tpc->cp0_y[idx_tri] = cp0.y;
		tpc->cp0_z[idx_tri] = cp0.z;
		tpc->cp1_x[idx_tri] = cp1.x;
		tpc->cp1_y[idx_tri] = cp1.y;
		tpc->cp1_z[idx_tri] = cp1.z;
		tpc->cp2_x[idx_tri] = cp2.x;
		tpc->cp2_y[idx_tri] = cp2.y;
		tpc->cp2_z[idx_tri] = cp2.z;

		tpc->rcp_dp2_v21[idx_tri] = 1.0f / length_sq(v21);
		tpc->rcp_dp2_v32[idx_tri] = 1.0f / length_sq(v32);
		tpc->rcp_dp2_v13[idx_tri] = 1.0f / length_sq(v13);

		tpc->rcp_dp2_nor[idx_tri] = 1.0f / length_sq(nor);
	}

	return tpc;
}

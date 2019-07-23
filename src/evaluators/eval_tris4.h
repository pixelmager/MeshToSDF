#pragma once

#include <sdf_support.h>
#include <evaluators/precalc.h>

// ====
__m128 udTriangle_sq_precalc_SIMD_soa_4tris( const __m128 p_x, const __m128 p_y, const __m128 p_z, tri_precalc_simd_soa_t const * const tpc, int idx_tri )
{
	const __m128 v1_x = _mm_load_ps(tpc->v1_x+idx_tri);
	const __m128 v1_y = _mm_load_ps(tpc->v1_y+idx_tri);
	const __m128 v1_z = _mm_load_ps(tpc->v1_z+idx_tri);

	const __m128 v2_x = _mm_load_ps(tpc->v2_x+idx_tri);
	const __m128 v2_y = _mm_load_ps(tpc->v2_y+idx_tri);
	const __m128 v2_z = _mm_load_ps(tpc->v2_z+idx_tri);

	const __m128 v3_x = _mm_load_ps(tpc->v3_x+idx_tri);
	const __m128 v3_y = _mm_load_ps(tpc->v3_y+idx_tri);
	const __m128 v3_z = _mm_load_ps(tpc->v3_z+idx_tri);

	//	const vec4_t p1 = sub(p, pc.v1);
	const __m128 p1_x = _mm_sub_ps( p_x, v1_x );
	const __m128 p1_y = _mm_sub_ps( p_y, v1_y );
	const __m128 p1_z = _mm_sub_ps( p_z, v1_z );

	//const vec4_t p2 = sub(p, pc.v2);
	const __m128 p2_x = _mm_sub_ps( p_x, v2_x );
	const __m128 p2_y = _mm_sub_ps( p_y, v2_y );
	const __m128 p2_z = _mm_sub_ps( p_z, v2_z );

	//const vec4_t p3 = sub(p, pc.v3);
	const __m128 p3_x = _mm_sub_ps( p_x, v3_x );
	const __m128 p3_y = _mm_sub_ps( p_y, v3_y );
	const __m128 p3_z = _mm_sub_ps( p_z, v3_z );

	const __m128 pc_v21_x = _mm_sub_ps( v2_x, v1_x );
	const __m128 pc_v21_y = _mm_sub_ps( v2_y, v1_y );
	const __m128 pc_v21_z = _mm_sub_ps( v2_z, v1_z );
	
	const __m128 pc_v32_x = _mm_sub_ps( v3_x, v2_x );
	const __m128 pc_v32_y = _mm_sub_ps( v3_y, v2_y );
	const __m128 pc_v32_z = _mm_sub_ps( v3_z, v2_z );
	
	const __m128 pc_v13_x = _mm_sub_ps( v1_x, v3_x );
	const __m128 pc_v13_y = _mm_sub_ps( v1_y, v3_y );
	const __m128 pc_v13_z = _mm_sub_ps( v1_z, v3_z );


	__m128 res0;
	{
		const __m128 zeros = _mm_setzero_ps();
		const __m128 ones  = _mm_set1_ps(1.0f);

		//vec3_t dp456;
		//dp456.x = dot(pc.v21, p1)
		// vec4_t mul012 = mul(dp456, pc.rcp_dp2_v21_v32_v13);
		// mul012.x = clamp( mul012.x, 0.0f, 1.0f );
		//
		// clamp(pc.v21.x*p1.x + pc.v21.y*p1.y + pc.v21.z*p1.z, 0, 1 );
		__m128 dp4;
		dp4 = _mm_mul_ps(   p1_x, pc_v21_x );
		dp4 = _mm_fmadd_ps( p1_y, pc_v21_y, dp4 );
		dp4 = _mm_fmadd_ps( p1_z, pc_v21_z, dp4 );
		dp4 = _mm_mul_ps(dp4, _mm_load_ps(tpc->rcp_dp2_v21+idx_tri) );
		dp4 = _mm_min_ps(dp4, ones);
		dp4 = _mm_max_ps(dp4, zeros);

		//dp456.y = dot(pc.v32, p2)
		__m128 dp5;
		dp5 = _mm_mul_ps(   p2_x, pc_v32_x );
		dp5 = _mm_fmadd_ps( p2_y, pc_v32_y, dp5 );
		dp5 = _mm_fmadd_ps( p2_z, pc_v32_z, dp5 );
		dp5 = _mm_mul_ps(dp5, _mm_load_ps(tpc->rcp_dp2_v32+idx_tri) );
		dp5 = _mm_min_ps(dp5, ones);
		dp5 = _mm_max_ps(dp5, zeros);

		//dp456.z = dot(pc.v13, p3)
		__m128 dp6;
		dp6 = _mm_mul_ps(   p3_x, pc_v13_x );
		dp6 = _mm_fmadd_ps( p3_y, pc_v13_y, dp6 );
		dp6 = _mm_fmadd_ps( p3_z, pc_v13_z, dp6 );
		dp6 = _mm_mul_ps(dp6, _mm_load_ps(tpc->rcp_dp2_v13+idx_tri) );
		dp6 = _mm_min_ps(dp6, ones);
		dp6 = _mm_max_ps(dp6, zeros);

		//vec4_t mul3 = mul(pc.v21, mul012.x);
		//vec4_t sub0 = sub(mul3, p1);
		__m128 sub0_x = _mm_fmsub_ps(pc_v21_x, dp4, p1_x);
		__m128 sub0_y = _mm_fmsub_ps(pc_v21_y, dp4, p1_y);
		__m128 sub0_z = _mm_fmsub_ps(pc_v21_z, dp4, p1_z);

		// vec4_t mul4 = mul(pc.v32, mul012.y);
		// vec4_t sub1 = sub(mul4, p2);
		__m128 sub1_x = _mm_fmsub_ps(pc_v32_x, dp5, p2_x);
		__m128 sub1_y = _mm_fmsub_ps(pc_v32_y, dp5, p2_y);
		__m128 sub1_z = _mm_fmsub_ps(pc_v32_z, dp5, p2_z);

		// vec4_t mul5 = mul(pc.v13, mul012.z);
		// vec4_t sub2 = sub(mul5, p3);
		__m128 sub2_x = _mm_fmsub_ps(pc_v13_x, dp6, p3_x);
		__m128 sub2_y = _mm_fmsub_ps(pc_v13_y, dp6, p3_y);
		__m128 sub2_z = _mm_fmsub_ps(pc_v13_z, dp6, p3_z);

		// float32_t len0 = dot( sub0, sub0 );
		__m128 lensq0;
		lensq0 = _mm_mul_ps(   sub0_x, sub0_x );
		lensq0 = _mm_fmadd_ps( sub0_y, sub0_y, lensq0 );
		lensq0 = _mm_fmadd_ps( sub0_z, sub0_z, lensq0 );

		// float32_t len1 = dot( sub1, sub1 )
		__m128 lensq1;
		lensq1 = _mm_mul_ps(   sub1_x, sub1_x );
		lensq1 = _mm_fmadd_ps( sub1_y, sub1_y, lensq1 );
		lensq1 = _mm_fmadd_ps( sub1_z, sub1_z, lensq1 );

		// float32_t len2 = dot( sub2, sub2 )
		__m128 lensq2;
		lensq2 = _mm_mul_ps(   sub2_x, sub2_x );
		lensq2 = _mm_fmadd_ps( sub2_y, sub2_y, lensq2 );
		lensq2 = _mm_fmadd_ps( sub2_z, sub2_z, lensq2 );

		//  float32_t f0 = min( min( len0, len1), len2 )
		res0 = _mm_min_ps( lensq0, lensq1 );
		res0 = _mm_min_ps( res0, lensq2 );
	}

	__m128 res1;
	{
		__m128 dp7;
		dp7 = _mm_mul_ps(   _mm_load_ps(tpc->nor_x+idx_tri), p1_x );
		dp7 = _mm_fmadd_ps( _mm_load_ps(tpc->nor_y+idx_tri), p1_y, dp7 );
		dp7 = _mm_fmadd_ps( _mm_load_ps(tpc->nor_z+idx_tri), p1_z, dp7 );

		res1 = _mm_mul_ps( dp7, dp7 );
		res1 = _mm_mul_ps( res1, _mm_load_ps(tpc->rcp_dp2_nor+idx_tri) );
	}

	// float32_t dp1 = dot(pc.cp0, p1)
	__m128 dp1;
	dp1 = _mm_mul_ps(   _mm_load_ps(tpc->cp0_x+idx_tri), p1_x );
	dp1 = _mm_fmadd_ps( _mm_load_ps(tpc->cp0_y+idx_tri), p1_y, dp1 );
	dp1 = _mm_fmadd_ps( _mm_load_ps(tpc->cp0_z+idx_tri), p1_z, dp1 );

	//float32_t dp2 = dot(pc.cp1, p2);
	__m128 dp2;
	dp2 = _mm_mul_ps(   _mm_load_ps(tpc->cp1_x+idx_tri), p2_x );
	dp2 = _mm_fmadd_ps( _mm_load_ps(tpc->cp1_y+idx_tri), p2_y, dp2 );
	dp2 = _mm_fmadd_ps( _mm_load_ps(tpc->cp1_z+idx_tri), p2_z, dp2 );

	//float32_t dp3 = dot(pc.cp2, p3);
	__m128 dp3;
	dp3 = _mm_mul_ps(   _mm_load_ps(tpc->cp2_x+idx_tri), p3_x );
	dp3 = _mm_fmadd_ps( _mm_load_ps(tpc->cp2_y+idx_tri), p3_y, dp3 );
	dp3 = _mm_fmadd_ps( _mm_load_ps(tpc->cp2_z+idx_tri), p3_z, dp3 );

	// float32_t s1 = sign(dp1);
	// float32_t s2 = sign(dp2);
	// float32_t s3 = sign(dp3);
	__m128 sign1 = sign(dp1);
	__m128 sign2 = sign(dp2);
	__m128 sign3 = sign(dp3);

	__m128 sum;
	sum = _mm_add_ps( sign1, sign2 );
	sum = _mm_add_ps( sum, sign3 );

	__m128 cmp = _mm_cmp_ps( sum, _mm_set1_ps(2.0f), _CMP_LT_OQ );

	__m128 res = _mm_blendv_ps( res1, res0, cmp );

	return res;
}


// ====
void eval_sdf__simd_soa_4tris( sdf_t &sdf, lpt::indexed_triangle_mesh_t const * const mesh )
{
	printf("%s\n", __FUNCTION__);

	aabb_t bb;
	bb.mn = vec3_t( sdf.header.bb_mn_x, sdf.header.bb_mn_y, sdf.header.bb_mn_z );
	bb.mx = vec3_t( sdf.header.bb_mx_x, sdf.header.bb_mx_y, sdf.header.bb_mx_z );
	vec3_t bb_range = bb.mx-bb.mn;

	tri_precalc_simd_soa_t * const tpc = precalc_simd_soa(mesh);

	const vec3_t stepsiz = vec3_t( bb_range.x / static_cast<float32_t>(sdf.header.dim_x),
								   bb_range.y / static_cast<float32_t>(sdf.header.dim_y),
								   bb_range.z / static_cast<float32_t>(sdf.header.dim_z) );
	
	const vec3_t p0 = bb.mn + 0.5f * stepsiz; //note: +0.5 to center in grid-cell

	const __m128 mm_stepsiz = _mm_set_ps( 0, stepsiz.z, stepsiz.y, stepsiz.x );
	const __m128 mm_p0 = _mm_set_ps(  1, p0.z, p0.y, p0.x );

	assert( mesh->tri_indices.size() % 3 == 0 );
	const int32_t num_tris = (int32_t)mesh->tri_indices.size()/3;

	for ( int z=0,zn=sdf.header.dim_z; z<zn; ++z )
	{
		for ( int y=0,yn=sdf.header.dim_y; y<yn; ++y )
		{
			for ( int x=0,xn=sdf.header.dim_x; x<xn; ++x )
			{
				const __m128 xyz = _mm_set_ps( 0, (float)z, (float)y, (float)x ); //TODO: _mm_add_ps( p, _mm_shuffle_ps(stepsiz, stepsiz, 0300)) instead?
				const __m128 mm_p = _mm_fmadd_ps( xyz, mm_stepsiz, mm_p0 );
		
				const __m128 p_x = SPLAT4_X(mm_p);
				const __m128 p_y = SPLAT4_Y(mm_p);
				const __m128 p_z = SPLAT4_Z(mm_p);

				__m128 d_min = _mm_set1_ps( FLT_MAX );
		
				const int32_t num_4tris = num_tris / 4;
				//const int32_t num_remain = num_tris - 4*num_4tris;
				for ( int32_t idx_tri=0; idx_tri<num_4tris; ++idx_tri )
				{
					d_min = _mm_min_ps( d_min, udTriangle_sq_precalc_SIMD_soa_4tris( p_x, p_y, p_z, tpc, 4*idx_tri ) );
				}
				////////////
				//TODO: eval remaining triangles... (roll back a bit, eval last 4, some again... doesn't matter as just min... does that make it unaligned?)
				////////////

				//TODO: unroll to 4 d_mins?
				int idx = x + y*xn + z*xn*yn;
				sdf.data[idx] = sqrtf(	min( d_min.m128_f32[0],
										min( d_min.m128_f32[1],
										min( d_min.m128_f32[2],
										     d_min.m128_f32[3] ))) );
			}
		}
	}

	_aligned_free( tpc );
}

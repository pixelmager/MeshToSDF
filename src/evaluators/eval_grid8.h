#pragma once

#include <sdf_support.h>
#include <evaluators/precalc.h>

// ====
__m256 udTriangle_sq_precalc_SIMD_8grid( const __m256 p_x, const __m256 p_y, const __m256 p_z, const tri_precalc_t &pc )
{
	const __m256 v1_x = _mm256_set1_ps(pc.v1.x);
	const __m256 v1_y = _mm256_set1_ps(pc.v1.y);
	const __m256 v1_z = _mm256_set1_ps(pc.v1.z);

	const __m256 v2_x = _mm256_set1_ps(pc.v2.x);
	const __m256 v2_y = _mm256_set1_ps(pc.v2.y);
	const __m256 v2_z = _mm256_set1_ps(pc.v2.z);

	const __m256 v3_x = _mm256_set1_ps(pc.v3.x);
	const __m256 v3_y = _mm256_set1_ps(pc.v3.y);
	const __m256 v3_z = _mm256_set1_ps(pc.v3.z);

	//	const vec4_t p1 = sub(p, pc.v1);
	const __m256 p1_x = _mm256_sub_ps( p_x, v1_x );
	const __m256 p1_y = _mm256_sub_ps( p_y, v1_y );
	const __m256 p1_z = _mm256_sub_ps( p_z, v1_z );

	//const vec4_t p2 = sub(p, pc.v2);
	const __m256 p2_x = _mm256_sub_ps( p_x, v2_x );
	const __m256 p2_y = _mm256_sub_ps( p_y, v2_y );
	const __m256 p2_z = _mm256_sub_ps( p_z, v2_z );

	//const vec4_t p3 = sub(p, pc.v3);
	const __m256 p3_x = _mm256_sub_ps( p_x, v3_x );
	const __m256 p3_y = _mm256_sub_ps( p_y, v3_y );
	const __m256 p3_z = _mm256_sub_ps( p_z, v3_z );


	const __m256 pc_v21_x = _mm256_sub_ps( v2_x, v1_x );
	const __m256 pc_v21_y = _mm256_sub_ps( v2_y, v1_y );
	const __m256 pc_v21_z = _mm256_sub_ps( v2_z, v1_z );
	
	const __m256 pc_v32_x = _mm256_sub_ps( v3_x, v2_x );
	const __m256 pc_v32_y = _mm256_sub_ps( v3_y, v2_y );
	const __m256 pc_v32_z = _mm256_sub_ps( v3_z, v2_z );
	
	const __m256 pc_v13_x = _mm256_sub_ps( v1_x, v3_x );
	const __m256 pc_v13_y = _mm256_sub_ps( v1_y, v3_y );
	const __m256 pc_v13_z = _mm256_sub_ps( v1_z, v3_z );


	__m256 res0;
	{
		const __m256 zeros = _mm256_setzero_ps();
		const __m256 ones  = _mm256_set1_ps(1.0f);

		//vec3_t dp456;
		//dp456.x = dot(pc.v21, p1)
		// vec4_t mul012 = mul(dp456, pc.rcp_dp2_v21_v32_v13);
		// mul012.x = clamp( mul012.x, 0.0f, 1.0f );
		//
		// clamp(pc.v21.x*p1.x + pc.v21.y*p1.y + pc.v21.z*p1.z, 0, 1 );
		__m256 dp4;
		dp4 = _mm256_mul_ps(   p1_x, pc_v21_x );
		dp4 = _mm256_fmadd_ps( p1_y, pc_v21_y, dp4 );
		dp4 = _mm256_fmadd_ps( p1_z, pc_v21_z, dp4 );
		dp4 = _mm256_mul_ps(dp4, _mm256_set1_ps(pc.rcp_dp2_v21_v32_v13.x) );
		dp4 = _mm256_min_ps(dp4, ones);
		dp4 = _mm256_max_ps(dp4, zeros);

		//dp456.y = dot(pc.v32, p2)
		__m256 dp5;
		dp5 = _mm256_mul_ps(   p2_x, pc_v32_x );
		dp5 = _mm256_fmadd_ps( p2_y, pc_v32_y, dp5 );
		dp5 = _mm256_fmadd_ps( p2_z, pc_v32_z, dp5 );
		dp5 = _mm256_mul_ps(dp5, _mm256_set1_ps(pc.rcp_dp2_v21_v32_v13.y) );
		dp5 = _mm256_min_ps(dp5, ones);
		dp5 = _mm256_max_ps(dp5, zeros);

		//dp456.z = dot(pc.v13, p3)
		__m256 dp6;
		dp6 = _mm256_mul_ps(   p3_x, pc_v13_x );
		dp6 = _mm256_fmadd_ps( p3_y, pc_v13_y, dp6 );
		dp6 = _mm256_fmadd_ps( p3_z, pc_v13_z, dp6 );
		dp6 = _mm256_mul_ps(dp6, _mm256_set1_ps(pc.rcp_dp2_v21_v32_v13.z) );
		dp6 = _mm256_min_ps(dp6, ones);
		dp6 = _mm256_max_ps(dp6, zeros);

		//vec4_t mul3 = mul(pc.v21, mul012.x);
		//vec4_t sub0 = sub(mul3, p1);
		__m256 sub0_x = _mm256_fmsub_ps(pc_v21_x, dp4, p1_x);
		__m256 sub0_y = _mm256_fmsub_ps(pc_v21_y, dp4, p1_y);
		__m256 sub0_z = _mm256_fmsub_ps(pc_v21_z, dp4, p1_z);

		// vec4_t mul4 = mul(pc.v32, mul012.y);
		// vec4_t sub1 = sub(mul4, p2);
		__m256 sub1_x = _mm256_fmsub_ps(pc_v32_x, dp5, p2_x);
		__m256 sub1_y = _mm256_fmsub_ps(pc_v32_y, dp5, p2_y);
		__m256 sub1_z = _mm256_fmsub_ps(pc_v32_z, dp5, p2_z);

		// vec4_t mul5 = mul(pc.v13, mul012.z);
		// vec4_t sub2 = sub(mul5, p3);
		__m256 sub2_x = _mm256_fmsub_ps(pc_v13_x, dp6, p3_x);
		__m256 sub2_y = _mm256_fmsub_ps(pc_v13_y, dp6, p3_y);
		__m256 sub2_z = _mm256_fmsub_ps(pc_v13_z, dp6, p3_z);

		// float32_t len0 = dot( sub0, sub0 );
		__m256 lensq0;
		lensq0 = _mm256_mul_ps(   sub0_x, sub0_x );
		lensq0 = _mm256_fmadd_ps( sub0_y, sub0_y, lensq0 );
		lensq0 = _mm256_fmadd_ps( sub0_z, sub0_z, lensq0 );

		// float32_t len1 = dot( sub1, sub1 )
		__m256 lensq1;
		lensq1 = _mm256_mul_ps(   sub1_x, sub1_x );
		lensq1 = _mm256_fmadd_ps( sub1_y, sub1_y, lensq1 );
		lensq1 = _mm256_fmadd_ps( sub1_z, sub1_z, lensq1 );

		// float32_t len2 = dot( sub2, sub2 )
		__m256 lensq2;
		lensq2 = _mm256_mul_ps(   sub2_x, sub2_x );
		lensq2 = _mm256_fmadd_ps( sub2_y, sub2_y, lensq2 );
		lensq2 = _mm256_fmadd_ps( sub2_z, sub2_z, lensq2 );

		//  float32_t f0 = min( min( len0, len1), len2 )
		res0 = _mm256_min_ps( lensq0, lensq1 );
		res0 = _mm256_min_ps( res0, lensq2 );
	}

	__m256 res1;
	{
		__m256 dp7;
		dp7 = _mm256_mul_ps(   _mm256_set1_ps(pc.nor.x), p1_x );
		dp7 = _mm256_fmadd_ps( _mm256_set1_ps(pc.nor.y), p1_y, dp7 );
		dp7 = _mm256_fmadd_ps( _mm256_set1_ps(pc.nor.z), p1_z, dp7 );

		res1 = _mm256_mul_ps( dp7, dp7 );
		res1 = _mm256_mul_ps( res1, _mm256_set1_ps(pc.rcp_dp2_nor) );
	}

	// float32_t dp1 = dot(pc.cp0, p1)
	__m256 dp1;
	dp1 = _mm256_mul_ps(   _mm256_set1_ps(pc.cp0.x), p1_x );
	dp1 = _mm256_fmadd_ps( _mm256_set1_ps(pc.cp0.y), p1_y, dp1 );
	dp1 = _mm256_fmadd_ps( _mm256_set1_ps(pc.cp0.z), p1_z, dp1 );

	//float32_t dp2 = dot(pc.cp1, p2);
	__m256 dp2;
	dp2 = _mm256_mul_ps(   _mm256_set1_ps(pc.cp1.x), p2_x );
	dp2 = _mm256_fmadd_ps( _mm256_set1_ps(pc.cp1.y), p2_y, dp2 );
	dp2 = _mm256_fmadd_ps( _mm256_set1_ps(pc.cp1.z), p2_z, dp2 );

	//float32_t dp3 = dot(pc.cp2, p3);
	__m256 dp3;
	dp3 = _mm256_mul_ps(   _mm256_set1_ps(pc.cp2.x), p3_x );
	dp3 = _mm256_fmadd_ps( _mm256_set1_ps(pc.cp2.y), p3_y, dp3 );
	dp3 = _mm256_fmadd_ps( _mm256_set1_ps(pc.cp2.z), p3_z, dp3 );

	// float32_t s1 = sign(dp1);
	// float32_t s2 = sign(dp2);
	// float32_t s3 = sign(dp3);
	__m256 sign1 = sign(dp1);
	__m256 sign2 = sign(dp2);
	__m256 sign3 = sign(dp3);

	__m256 sum;
	sum = _mm256_add_ps( sign1, sign2 );
	sum = _mm256_add_ps( sum, sign3 );

	__m256 cmp = _mm256_cmp_ps( sum, _mm256_set1_ps(2.0f), _CMP_LT_OQ );
	__m256 res = _mm256_blendv_ps( res1, res0, cmp );

	return res;
}

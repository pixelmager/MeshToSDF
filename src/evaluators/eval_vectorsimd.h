#pragma once

#include <sdf_support.h>
#include <evaluators/precalc.h>

// ====
//TODO struct { __m128 closest_distance; __m128 closest_tri_idx; };
__m128 udTriangle_sq_precalc_SIMD_aos( const __m128 mm_p, const tri_precalc_simd_aos_t &pc )
{
	const __m128 mm_p1 = _mm_sub_ps(mm_p, pc.v1);
	const __m128 mm_p2 = _mm_sub_ps(mm_p, pc.v2);
	const __m128 mm_p3 = _mm_sub_ps(mm_p, pc.v3);

	__m128 res0;
	{
		__m128 mm_dp4 = _mm_dp_ps( pc.v21, mm_p1, 0xF1);
		__m128 mm_dp5 = _mm_dp_ps( pc.v32, mm_p2, 0xF1);
		__m128 mm_dp6 = _mm_dp_ps( pc.v13, mm_p3, 0xF1);

		__m128 mm_dp456 = _mm_shuffle_ps( mm_dp4, mm_dp5, _MM_SHUFFLE(0,0,0,0) );
		mm_dp456 = _mm_shuffle_ps( mm_dp456, mm_dp6, _MM_SHUFFLE(2,0,2,0) );

		__m128 mm_mul012 = _mm_mul_ps( mm_dp456, pc.rcp_dp2_v21_v32_v13 );
		mm_mul012 = _mm_min_ps(mm_mul012, _mm_set1_ps(1.0f));
		mm_mul012 = _mm_max_ps(mm_mul012, _mm_setzero_ps());

		__m128 mm_mul012_x = _mm_shuffle_ps( mm_mul012, mm_mul012, _MM_SHUFFLE(0,0,0,0) ); //note: c*vec
		__m128 mm_mul012_y = _mm_shuffle_ps( mm_mul012, mm_mul012, _MM_SHUFFLE(1,1,1,1) );
		__m128 mm_mul012_z = _mm_shuffle_ps( mm_mul012, mm_mul012, _MM_SHUFFLE(2,2,2,2) );

		__m128 mm_sub0 = _mm_fmsub_ps(pc.v21, mm_mul012_x, mm_p1);
		__m128 mm_sub1 = _mm_fmsub_ps(pc.v32, mm_mul012_y, mm_p2);
		__m128 mm_sub2 = _mm_fmsub_ps(pc.v13, mm_mul012_z, mm_p3);

		__m128 mm_lensq0 = _mm_dp_ps( mm_sub0, mm_sub0, 0xF1 );
		__m128 mm_lensq1 = _mm_dp_ps( mm_sub1, mm_sub1, 0xF1 );
		__m128 mm_lensq2 = _mm_dp_ps( mm_sub2, mm_sub2, 0xF1 );

		res0 = _mm_min_ss( mm_lensq0, mm_lensq1 ); //note:f32[0]
		res0 = _mm_min_ss( res0, mm_lensq2 ); //note:f32[0]
	}
	
	__m128 res1;
	{
		__m128 mm_dp7 = _mm_dp_ps(pc.nor, mm_p1, 0xF1);
		res1 = _mm_mul_ss(mm_dp7, mm_dp7);
		res1 = _mm_mul_ss( res1, pc.rcp_dp2_nor );
	}

	__m128 mm_dp1 = _mm_dp_ps(pc.cp0, mm_p1, 0xF1);
	__m128 mm_dp2 = _mm_dp_ps(pc.cp1, mm_p2, 0xF1);
	__m128 mm_dp3 = _mm_dp_ps(pc.cp2, mm_p3, 0xF1);
	__m128 mm_dp123 = _mm_shuffle_ps( mm_dp1, mm_dp2, _MM_SHUFFLE(0,0,0,0) );
	mm_dp123 = _mm_shuffle_ps( mm_dp123, mm_dp3, _MM_SHUFFLE(2,0,2,0) );

	struct hz_sum_ps
	{
		//note: from https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-float-vector-sum-on-x86
		static inline __m128 eval( const __m128 v )
		{
			//TODO: https://deplinenoise.files.wordpress.com/2015/03/gdc2015_afredriksson_simd.pdf
			__m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
			__m128 sums = _mm_add_ps(v, shuf);
			shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
			sums        = _mm_add_ss(sums, shuf);
			return sums;
		}
	};

	__m128 s123 = sign( mm_dp123 );
	__m128 sum = hz_sum_ps::eval(s123); //note: f32[0]
	__m128 cmp = _mm_cmplt_ss( sum, _mm_set1_ps(2.0f) ); //note: f32[0]
	return _mm_blendv_ps( res1, res0, cmp); //note: f32[0]
}

// ====
void eval_sdf__precalc_simd_aos( sdf_t &sdf, lpt::indexed_triangle_mesh_t const * const mesh )
{
	printf("%s\n", __FUNCTION__);
	aabb_t bb;
	bb.mn = vec3_t( sdf.header.bb_mn_x, sdf.header.bb_mn_y, sdf.header.bb_mn_z );
	bb.mx = vec3_t( sdf.header.bb_mx_x, sdf.header.bb_mx_y, sdf.header.bb_mx_z );
	vec3_t bb_range = bb.mx-bb.mn;

	tri_precalc_simd_aos_t * const tpc = precalc_simd_aos(mesh);

	const vec3_t stepsiz = vec3_t( bb_range.x / static_cast<float32_t>(sdf.header.dim_x),
								   bb_range.y / static_cast<float32_t>(sdf.header.dim_y),
								   bb_range.z / static_cast<float32_t>(sdf.header.dim_z) );
	
	const vec3_t p0 = bb.mn - 0.5f * stepsiz; //note: -0.5, because +0.5*stepsize to center at cell, -stepsiz for loop-init


	vec3_t p = p0;

	for ( int z=0,zn=sdf.header.dim_z; z<zn; ++z )
	{
		//printf( "z=%d\n", z );
		p.z += stepsiz.z;
		
		for ( int y=0,yn=sdf.header.dim_y; y<yn; ++y )
		{
			p.y += stepsiz.y;
		
			for ( int x=0,xn=sdf.header.dim_x; x<xn; ++x )
			{
				p.x += stepsiz.x;
		
				#if defined( DEBUG_LEVEL_PARANOID )
				vec3_t p_nm = vec3_t( (static_cast<float32_t>(x)+0.5f) / static_cast<float32_t>(xn),
									  (static_cast<float32_t>(y)+0.5f) / static_cast<float32_t>(yn),
									  (static_cast<float32_t>(z)+0.5f) / static_cast<float32_t>(zn) );
				vec3_t ps = bb.mn + bb_range * p_nm;
				assert( p.x == ps.x && p.y == ps.y && p.z == ps.z );
				#endif //DEBUG_LEVEL_PARANOID
		
				__m128 mm_p = _mm_load_ps( vec4_t(p,1.0f).ptr() );
				__m128 d_min = _mm_set1_ps( FLT_MAX );
		
				assert( mesh->tri_indices.size() % 3 == 0 );
				for ( size_t idx_tri=0,num_tris=mesh->tri_indices.size()/3; idx_tri<num_tris; ++idx_tri )
				{
					d_min = _mm_min_ps( d_min, udTriangle_sq_precalc_SIMD_aos( mm_p, tpc[idx_tri] ) );
				}
				int idx = x + y*xn + z*xn*yn;
				sdf.data[idx] = sqrtf( d_min.m128_f32[0] );
			}
			p.x = p0.x;
		}
		p.y = p0.y;
	}

	_aligned_free( tpc );
}


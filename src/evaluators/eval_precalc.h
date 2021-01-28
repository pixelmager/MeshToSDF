#pragma once

#include <sdf_support.h>
#include <evaluators/precalc.h>

float32_t udTriangle_sq_precalc( const vec3_t &p, const tri_precalc_t &pc )
{
    PROFILE_FUNC();

	const vec3_t p1 = p - pc.v1;
	const vec3_t p2 = p - pc.v2;
	const vec3_t p3 = p - pc.v3;

	return ((sign(dot(pc.cp0,p1)) + 
             sign(dot(pc.cp1,p2)) + 
             sign(dot(pc.cp2,p3))<2.0f) 
             ?
             min( min( 
             dot2(pc.v21*clamp(dot(pc.v21,p1)*pc.rcp_dp2_v21_v32_v13.x,0.0f,1.0f)-p1),
             dot2(pc.v32*clamp(dot(pc.v32,p2)*pc.rcp_dp2_v21_v32_v13.y,0.0f,1.0f)-p2) ),
             dot2(pc.v13*clamp(dot(pc.v13,p3)*pc.rcp_dp2_v21_v32_v13.z,0.0f,1.0f)-p3) )
             :
             dot(pc.nor,p1)*dot(pc.nor,p1)*pc.rcp_dp2_nor );
}
void eval_sdf__precalc( sdf_t &sdf, lpt::indexed_triangle_mesh_t const * const mesh )
{
	printf("%s\n", __FUNCTION__);
	aabb_t bb;
	bb.mn = vec3_t( sdf.header.bb_mn_x, sdf.header.bb_mn_y, sdf.header.bb_mn_z );
	bb.mx = vec3_t( sdf.header.bb_mx_x, sdf.header.bb_mx_y, sdf.header.bb_mx_z );
	vec3_t bb_range = bb.mx-bb.mn;

	tri_precalc_t * const tpc = precalc_tridata( mesh );

	for ( int z=0,zn=sdf.header.dim_z; z<zn; ++z ) {
	printf( "z=%d\n", z );
	for ( int y=0,yn=sdf.header.dim_y; y<yn; ++y ) {
	for ( int x=0,xn=sdf.header.dim_x; x<xn; ++x )
	{
		float32_t d_min =  FLT_MAX;

		vec3_t p_nm = vec3_t( (static_cast<float32_t>(x)+0.5f) / static_cast<float32_t>(xn),
							  (static_cast<float32_t>(y)+0.5f) / static_cast<float32_t>(yn),
							  (static_cast<float32_t>(z)+0.5f) / static_cast<float32_t>(zn) );
		vec3_t p = bb.mn + bb_range * p_nm;

		//TODO: alternatively calculate few tris for a bunch of grid-cells to fetch less often?
		assert( mesh->tri_indices.size() % 3 == 0 );
		for ( size_t idx_tri=0,num_tris=mesh->tri_indices.size()/3; idx_tri<num_tris; ++idx_tri )
		{
			float32_t ud = udTriangle_sq_precalc( p, tpc[idx_tri] );

			if ( ud < d_min )
				d_min = ud;
		}
		int idx = x + y*xn + z*xn*yn;
		sdf.data[idx] = sqrtf( d_min );
	}}}

	_aligned_free( tpc );
}


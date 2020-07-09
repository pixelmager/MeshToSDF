#pragma once

#include <sdf_support.h>
#include <evaluators/precalc.h>

///////////////////////////////////////////////////////////////////////////////

//note: from http://iquilezles.org/www/articles/triangledistance/triangledistance.htm
float32_t dot2( const vec3_t &v ) { return dot(v,v); }

float32_t udTriangle( const vec3_t &v1, const vec3_t &v2, const vec3_t &v3, const vec3_t &p )
{
    vec3_t v21 = v2 - v1; vec3_t p1 = p - v1;
    vec3_t v32 = v3 - v2; vec3_t p2 = p - v2;
    vec3_t v13 = v1 - v3; vec3_t p3 = p - v3;
    vec3_t nor = cross( v21, v13 );

    return sqrtf((sign(dot(cross(v21,nor),p1)) + 
             sign(dot(cross(v32,nor),p2)) + 
             sign(dot(cross(v13,nor),p3))<2.0f) 
             ?
             min( min( 
             dot2(v21*clamp(dot(v21,p1)/dot2(v21),0.0f,1.0f)-p1),
             dot2(v32*clamp(dot(v32,p2)/dot2(v32),0.0f,1.0f)-p2) ),
             dot2(v13*clamp(dot(v13,p3)/dot2(v13),0.0f,1.0f)-p3) )
             :
             dot(nor,p1)*dot(nor,p1)/dot2(nor) );
}
float32_t udTriangle_sq( const vec3_t &v1, const vec3_t &v2, const vec3_t &v3, const vec3_t &p )
{
    vec3_t v21 = v2 - v1; vec3_t p1 = p - v1;
    vec3_t v32 = v3 - v2; vec3_t p2 = p - v2;
    vec3_t v13 = v1 - v3; vec3_t p3 = p - v3;
    vec3_t nor = cross( v21, v13 );

	return ((sign(dot(cross(v21,nor),p1)) + 
             sign(dot(cross(v32,nor),p2)) + 
             sign(dot(cross(v13,nor),p3))<2.0f) 
             ?
             min( min( 
             dot2(v21*clamp(dot(v21,p1)/dot2(v21),0.0f,1.0f)-p1),
             dot2(v32*clamp(dot(v32,p2)/dot2(v32),0.0f,1.0f)-p2) ),
             dot2(v13*clamp(dot(v13,p3)/dot2(v13),0.0f,1.0f)-p3) )
             :
             dot(nor,p1)*dot(nor,p1)/dot2(nor) );
}

// ====
void eval_sdf__bruteforce( sdf_t &sdf, lpt::indexed_triangle_mesh_t const * const mesh )
{
	printf("%s\n", __FUNCTION__);
	assert( mesh->positions.size() % 3 == 0 );
	vec3_t const * const positions = reinterpret_cast<vec3_t const * const>( &mesh->positions[0] );

	aabb_t bb;
	bb.mn = vec3_t( sdf.header.bb_mn_x, sdf.header.bb_mn_y, sdf.header.bb_mn_z );
	bb.mx = vec3_t( sdf.header.bb_mx_x, sdf.header.bb_mx_y, sdf.header.bb_mx_z );
	vec3_t bb_range = bb.mx-bb.mn;

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

		assert( mesh->tri_indices.size() % 3 == 0 );
		for ( size_t idx_tri=0,num_tris=mesh->tri_indices.size()/3; idx_tri<num_tris; ++idx_tri )
		{
			const uint32_t idx0 = mesh->tri_indices[ 3*idx_tri+0 ];
			const uint32_t idx1 = mesh->tri_indices[ 3*idx_tri+1 ];
			const uint32_t idx2 = mesh->tri_indices[ 3*idx_tri+2 ];
			const vec3_t &p0 = positions[ idx0 ];
			const vec3_t &p1 = positions[ idx1 ];
			const vec3_t &p2 = positions[ idx2 ];

			//float32_t ud = udTriangle( p0, p1, p2, p );
			float32_t ud = udTriangle_sq( p0, p1, p2, p );

			if ( ud < d_min )
				d_min = ud;
		}
		int idx = x + y*xn + z*xn*yn;
		//sdf.data[idx] = d_min;
		sdf.data[idx] = sqrtf( d_min );

		#ifndef NDEBUG
		sdf.eval_points[ idx ] = p;
		#endif //NDEBUG
	}}}
}

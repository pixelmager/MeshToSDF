using System.Collections;
using System.Collections.Generic;

using UnityEngine;
using Unity.Burst;
using Unity.Mathematics;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class mesh2sdf : MonoBehaviour
{
    [UnityEditor.MenuItem("Custom/Convert Mesh to SDF")]
    public static void CalcSDF()
    {
        if ( UnityEditor.Selection.activeGameObject == null )
        {
            Debug.Log( "select object with mesh" );
        }

        GameObject go = UnityEditor.Selection.activeGameObject;
        MeshFilter mf = go.GetComponent<MeshFilter>();

    }

    // ======================================================================================
    #define SDF_PRECALC
    #if SDF_PRECALC
    struct tri_precalc_t
    {
	    float3 v1;
	    float3 v2;
	    float3 v3;
	    float3 v21;
	    float3 v32;
	    float3 v13;

	    float3 nor;
	    float3 cp0;
	    float3 cp1;
	    float3 cp2;

	    float3 rcp_dp2_v21_v32_v13;

	    float32_t rcp_dp2_nor;
    };

    tri_precalc_t* precalc_tridata( lpt::indexed_triangle_mesh_t const * const mesh )
    {
	    PROFILE_FUNC();
	
	    tri_precalc_t * const tpc = (tri_precalc_t*)_aligned_malloc( sizeof(tri_precalc_t) * (mesh->tri_indices.size() / 3 ), 16 );

	    assert( mesh->positions.size() % 3 == 0 );
	    float3 const * const positions = reinterpret_cast<float3 const * const>( &mesh->positions[0] );

	    for ( size_t idx_tri=0,num_tris=mesh->tri_indices.size()/3; idx_tri<num_tris; ++idx_tri )
	    {
		    const uint32_t idx0 = mesh->tri_indices[ 3*idx_tri+0 ];
		    const uint32_t idx1 = mesh->tri_indices[ 3*idx_tri+1 ];
		    const uint32_t idx2 = mesh->tri_indices[ 3*idx_tri+2 ];
		    const float3 &pos_v1 = positions[ idx0 ];
		    const float3 &pos_v2 = positions[ idx1 ];
		    const float3 &pos_v3 = positions[ idx2 ];

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

		    pc.rcp_dp2_v21_v32_v13 = float3(1.0f / length_sq(pc.v21),
										    1.0f / length_sq(pc.v32),
										    1.0f / length_sq(pc.v13) );

		    pc.rcp_dp2_nor = 1.0f / length_sq(pc.nor);
	    }

	    return tpc;
    }


    float udTriangle_sq_precalc( float3 p, tri_precalc_t pc )
    {
	    float3 p1 = p - pc.v1;
        float3 p2 = p - pc.v2;
        float3 p3 = p - pc.v3;

	    return ((sign(dot(pc.cp0, p1)) + 
                 sign(dot(pc.cp1, p2)) + 
                 sign(dot(pc.cp2, p3))<2.0f) 
                 ?
                 min(min(
                 dot2(pc.v21* clamp(dot(pc.v21, p1) * pc.rcp_dp2_v21_v32_v13.x,0.0f,1.0f)-p1),
                 dot2(pc.v32* clamp(dot(pc.v32, p2) * pc.rcp_dp2_v21_v32_v13.y,0.0f,1.0f)-p2) ),
                 dot2(pc.v13* clamp(dot(pc.v13, p3) * pc.rcp_dp2_v21_v32_v13.z,0.0f,1.0f)-p3) )
                 :
                 dot(pc.nor, p1)*dot(pc.nor, p1)*pc.rcp_dp2_nor );
    }
    void eval_sdf__precalc(sdf_t &sdf, lpt::indexed_triangle_mesh_t const * const mesh )
    {
        //printf("%s\n", __FUNCTION__);
        aabb_t bb;
        bb.mn = float3(sdf.header.bb_mn_x, sdf.header.bb_mn_y, sdf.header.bb_mn_z);
        bb.mx = float3(sdf.header.bb_mx_x, sdf.header.bb_mx_y, sdf.header.bb_mx_z);
        float3 bb_range = bb.mx - bb.mn;

        tri_precalc_t * const tpc = precalc_tridata(mesh);

        for (int z = 0, zn = sdf.header.dim_z; z < zn; ++z)
        {
            printf("z=%d\n", z);
            for (int y = 0, yn = sdf.header.dim_y; y < yn; ++y)
            {
                for (int x = 0, xn = sdf.header.dim_x; x < xn; ++x)
                {
                    float32_t d_min = FLT_MAX;

                    float3 p_nm = float3((static_cast<float32_t>(x) + 0.5f) / static_cast<float32_t>(xn),
                                          (static_cast<float32_t>(y) + 0.5f) / static_cast<float32_t>(yn),
                                          (static_cast<float32_t>(z) + 0.5f) / static_cast<float32_t>(zn));
                    float3 p = bb.mn + bb_range * p_nm;

                    //TODO: alternatively calculate few tris for a bunch of grid-cells to fetch less often?
                    assert(mesh->tri_indices.size() % 3 == 0);
                    for (size_t idx_tri = 0, num_tris = mesh->tri_indices.size() / 3; idx_tri < num_tris; ++idx_tri)
                    {
                        float32_t ud = udTriangle_sq_precalc(p, tpc[idx_tri]);

                        if (ud < d_min)
                            d_min = ud;
                    }
                    int idx = x + y * xn + z * xn * yn;
                    sdf.data[idx] = sqrtf(d_min);
                }
            }
        }

        _aligned_free(tpc);
    }
    #endif //SDF_PRECALC
}

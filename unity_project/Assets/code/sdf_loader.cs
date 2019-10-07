using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class sdf_loader
{
    public static Material mat;

    struct header_t
    {
        public System.Int32 dim_x, dim_y, dim_z;
        public System.Single bb_mn_x, bb_mn_y, bb_mn_z;
        public System.Single bb_mx_x, bb_mx_y, bb_mx_z;
    };
    struct sdf_t
    {
        public header_t header;
        public System.Single[] data;
    };

    [UnityEditor.MenuItem("Custom/Convert SDF")]
    public static void Convert_SDF()
    {
        string fn = "Assets/tex3D_raw/sdf.bin";

        Debug.Log("loading \"" + fn + "\"" );

        sdf_t sdf = new sdf_t();

        using (System.IO.BinaryReader br = new System.IO.BinaryReader(System.IO.File.Open(fn, System.IO.FileMode.Open)))
        {
            sdf.header.dim_x = br.ReadInt32();
            sdf.header.dim_y = br.ReadInt32();
            sdf.header.dim_z = br.ReadInt32();

            sdf.header.bb_mn_x = br.ReadSingle();
            sdf.header.bb_mn_y = br.ReadSingle();
            sdf.header.bb_mn_z = br.ReadSingle();

            sdf.header.bb_mx_x = br.ReadSingle();
            sdf.header.bb_mx_y = br.ReadSingle();
            sdf.header.bb_mx_z = br.ReadSingle();

            Debug.Log( "header: ["
                + sdf.header.dim_x + ","
                + sdf.header.dim_y + ","
                + sdf.header.dim_z + "], bbmin=("
                + sdf.header.bb_mn_x + ","
                + sdf.header.bb_mn_y + ","
                + sdf.header.bb_mn_z + ") bbmax=("
                + sdf.header.bb_mx_x + ","
                + sdf.header.bb_mx_y + ","
                + sdf.header.bb_mx_z + ")" );

            int num_elems = sdf.header.dim_x * sdf.header.dim_y * sdf.header.dim_z;
            sdf.data = new System.Single[ num_elems ];
            for ( int i=0, n = num_elems; i<n; ++i )
            {
                sdf.data[i] = br.ReadSingle();
            }
        }

        int num_elems2 = sdf.header.dim_x * sdf.header.dim_y * sdf.header.dim_z; //ffs2

        float minval = float.MaxValue;
        float maxval = float.MinValue;
        int minidx = -1;
        int maxidx = -1;
        for (int i = 0, n = num_elems2; i < n; ++i)
        {
            float d = sdf.data[i];
            if ( d < minval )
            {
                minidx = i;
                minval = d;
            }
            if ( d > maxval ) 
            {
                maxidx = i;
                maxval = d;
            }
        }

        Debug.Log( "data_min: " + minval + "(" + minidx + ") data_max: " + maxval + "(" + maxidx + ")" );

        Color[] colors = new Color[ num_elems2 ];
        for ( int i=0, n=num_elems2; i<n; ++i )
        {
            float d = sdf.data[i];
            d = (d-minval) / (maxval-minval); //note: normalize to [0;1]
            colors[i] = new Color( d, d, d, d );
        }

        Texture3D densitymap_a8 = new Texture3D(sdf.header.dim_x, sdf.header.dim_y, sdf.header.dim_z, TextureFormat.Alpha8, mipmap: false);
        densitymap_a8.wrapMode = TextureWrapMode.Clamp;
        densitymap_a8.filterMode = FilterMode.Point;
        densitymap_a8.mipMapBias = 0;
        densitymap_a8.anisoLevel = 0;

        densitymap_a8.SetPixels( colors );
        densitymap_a8.Apply(updateMipmaps: false);

        AssetDatabase.CreateAsset(densitymap_a8, "Assets/tex3D/TEST_tex3D_density01_a8.asset");
        //TODO: load asset again?

        Material mat = AssetDatabase.LoadAssetAtPath<Material>("Assets/materials/mat_vis3dtex.mat");
        if ( mat != null )
        {
            mat.SetTexture("_MainTex", densitymap_a8 );
        }
    }
}

Shader "Unlit/vis_3dtex"
{
	Properties
	{
		_VolumeTex("VolumeTexture", 3D) = "white" {}
	}
	SubShader
	{
		Pass
		{
			Blend One One
			ZWrite Off

			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			
			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float2 texcoord : TEXCOORD0;
			};

			struct v2f
			{
				float4 vertex : SV_POSITION;
				float3 os_pos : TEXCOORD0;
			};

			float _Offset;

			sampler3D _VolumeTex;
			
			v2f vert (appdata v)
			{
				v2f o;
				float4 os_pos = v.vertex;
				o.vertex = UnityObjectToClipPos(os_pos);
				o.os_pos = float3( v.texcoord, UNITY_MATRIX_M._m13 /10.0f );
				return o;
			}
			
			fixed4 frag (v2f i) : SV_Target
			{
				fixed4 col = 0.5 * tex3D( _VolumeTex, i.os_pos ).aaaa;
				return col;
			}
			ENDCG
		}
	}
}

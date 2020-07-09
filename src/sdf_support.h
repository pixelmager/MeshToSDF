#pragma once 

#include <cstdio>
#include <float.h>
#include <assert.h>
#include <cmath>

#include <thread>
#include <ppl.h>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

//#include <intrin.h>
#include <immintrin.h>
//#include <zmmintrin.h>

#include <profiler.h>

#ifndef int32_t
typedef int int32_t;
#endif

#ifndef uint32_t
typedef unsigned int uint32_t;
#endif

#ifndef float32_t
typedef float float32_t;
#endif

///////////////////////////////////////////////////////////////////////////////

struct vec3_t
{
	float32_t x, y, z;
	vec3_t(){}
	vec3_t( float32_t in_x, float32_t in_y, float32_t in_z ) : x(in_x), y(in_y), z(in_z) {}
	vec3_t( const vec3_t &v ) : x(v.x), y(v.y), z(v.z) {}
	explicit vec3_t( __m128 v ) : x(v.m128_f32[0]), y(v.m128_f32[1]), z(v.m128_f32[2] ) {}
};

float32_t dot( const vec3_t &a, const vec3_t &b )
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}
vec3_t operator+( const vec3_t &a, const vec3_t &b )
{
	return vec3_t( a.x + b.x,
				   a.y + b.y,
				   a.z + b.z );
}
vec3_t sub( const vec3_t &a, const vec3_t &b )
{
	return vec3_t( a.x - b.x,
				a.y - b.y,
				a.z - b.z );
}
vec3_t operator-( const vec3_t &a, const vec3_t &b )
{
	return sub(a, b);
}
vec3_t mul( const vec3_t &a, const vec3_t &b )
{
	return vec3_t(  a.x * b.x,
					a.y * b.y,
					a.z * b.z );
}
vec3_t operator*( const vec3_t &a, const vec3_t &b )
{
	return mul(a, b);
}
vec3_t operator*( const vec3_t &v, const float32_t c )
{
	return vec3_t( v.x * c,
				   v.y * c,
				   v.z * c );
}
vec3_t operator*( const float32_t c, const vec3_t &v )
{
	return vec3_t( v.x * c,
				   v.y * c,
				   v.z * c );
}

vec3_t cross( const vec3_t &a, const vec3_t &b )
{
	return vec3_t( a.y*b.z - a.z*b.y,
				   a.z*b.x - a.x*b.z,
				   a.x*b.y - a.y*b.x );
}

////////////////////////////////////////////////

struct vec4_t
{
	float32_t x, y, z, w;

	vec4_t(){}
	vec4_t( float32_t in_x, float32_t in_y, float32_t in_z, float32_t in_w ) : x(in_x), y(in_y), z(in_z), w(in_w) {}
	vec4_t( const vec3_t &v, float in_w ) : x(v.x), y(v.y), z(v.z), w(in_w) {}
	vec4_t( const vec4_t &v ) : x(v.x), y(v.y), z(v.z), w(v.w) {}
	vec3_t xyz()
	{
		return vec3_t(x,y,z);
	}
	float32_t const * const ptr() const
	{
		return &x;
	}
};

float32_t dot( const vec4_t &a, const vec4_t &b )
{
	return a.x*b.x
		 + a.y*b.y
		 + a.z*b.z
		 + a.w*b.w;
}
vec4_t add( const vec4_t &a, const vec4_t &b )
{
	return vec4_t( a.x + b.x,
				   a.y + b.y,
				   a.z + b.z,
				   a.w + b.w );
}
vec4_t operator+( const vec4_t &a, const vec4_t &b )
{
	return add( a, b );
}
vec4_t sub( const vec4_t &a, const vec4_t &b )
{
	return vec4_t( a.x - b.x,
				   a.y - b.y,
				   a.z - b.z,
				   a.w - b.w );
}
vec4_t operator-( const vec4_t &a, const vec4_t &b )
{
	return sub(a, b);
}
vec4_t mul( const vec4_t &a, const vec4_t &b )
{
	return vec4_t(  a.x * b.x,
					a.y * b.y,
					a.z * b.z,
					a.w * b.w );
}
vec4_t mul( const vec4_t &v, const float32_t &c )
{
		return vec4_t( v.x * c,
					   v.y * c,
					   v.z * c,
					   v.w * c);
}
vec4_t mul( const float32_t &c, const vec4_t &v )
{
	return mul( v, c );
}
vec4_t operator*( const vec4_t &a, const vec4_t &b )
{
	return mul( a, b );
}
vec4_t operator*( const vec4_t &v, const float32_t c )
{
	return mul( v, c );
}
vec4_t operator*( const float32_t c, const vec4_t &v )
{
	return mul( v, c );
}

////////////////////////////////////////////////

namespace lpt
{
	inline float32_t sign( const float32_t v )
	{
		//TODO: return sign bit?
		//return (v>=0.0f) ? 1.0f : -1.0f;

		// https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/sign.xhtml
		if ( v == 0.0f )
			return 0.0f;
		return (v>0.0f) ? 1.0f : -1.0f;
	}

	int32_t min( int32_t a, int32_t b )
	{
		return (a<b) ? a : b;
	}
	float32_t min( float32_t a, float32_t b )
	{
		return (a<b) ? a : b;
	}
	vec3_t min( const vec3_t &v0, const vec3_t &v1 )
	{
		return vec3_t( min(v0.x, v1.x),
					   min(v0.y, v1.y),
					   min(v0.z, v1.z) );
	}

	float32_t max( float32_t a, float32_t b )
	{
		return (a>b) ? a : b;
	}
	vec3_t max( const vec3_t &v0, const vec3_t &v1 )
	{
		return vec3_t( max(v0.x, v1.x),
					   max(v0.y, v1.y),
					   max(v0.z, v1.z) );
	}
	float32_t clamp( float32_t v, float32_t mn, float32_t mx )
	{
		return max(mn, min(mx,v));
	}
	float32_t saturate( float32_t v )
	{
		return clamp( v, 0, 1 );
	}

	float32_t length_sq( const vec3_t &v )
	{
		return dot(v,v);
	}
	float32_t length( const vec3_t &v )
	{
		return sqrtf( length_sq(v) );
	}

	float32_t length_sq( const vec4_t &v )
	{
		return dot(v,v);
	}

	//note: from https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
	bool AlmostEqualRelative(float A, float B, float maxRelDiff = FLT_EPSILON)
	{
		// Calculate the difference.
		float diff = fabs(A - B);
		A = fabs(A);
		B = fabs(B);
		// Find the largest
		float largest = (B > A) ? B : A;
 
		if (diff <= largest * maxRelDiff)
			return true;
		return false;
	}
} //namespace lpt

///////////////////////////////////////////////////////////////////////////////

#define ALIGN16 _declspec(align(16))

//#define SPLAT4_X( V ) _mm_shuffle_ps((V),(V),_MM_SHUFFLE(0,0,0,0))
//#define SPLAT4_Y( V ) _mm_shuffle_ps((V),(V),_MM_SHUFFLE(1,1,1,1))
//#define SPLAT4_Z( V ) _mm_shuffle_ps((V),(V),_MM_SHUFFLE(2,2,2,2))

//#define SPLAT16_X(V) _mm512_maskz_broadcastss_ps(0xffffU, _mm_shuffle_ps((V),(V),_MM_SHUFFLE(0,0,0,0)) )
//#define SPLAT16_Y(V) _mm512_maskz_broadcastss_ps(0xffffU, _mm_shuffle_ps((V),(V),_MM_SHUFFLE(1,1,1,1)) )
//#define SPLAT16_Z(V) _mm512_maskz_broadcastss_ps(0xffffU, _mm_shuffle_ps((V),(V),_MM_SHUFFLE(2,2,2,2)) )

// ====
//note: from https://stackoverflow.com/questions/41315420/how-to-implement-sign-function-with-sse3
//inline __m128 sign_old( const __m128 x )
//{
//	__m128 zeros = _mm_setzero_ps();
//    __m128 positive = _mm_and_ps(_mm_cmpgt_ps(x, zeros), _mm_set1_ps(1.0f));
//    __m128 negative = _mm_and_ps(_mm_cmplt_ps(x, zeros), _mm_set1_ps(-1.0f));
//
//    return _mm_or_ps(positive, negative);
//}

//note: reasonable approximation, about 4% faster, 1.2% error
//inline __m128 sign_fast( const __m128 x )
//{
//	return _mm_blendv_ps( _mm_set1_ps(-1.0f), _mm_set1_ps(1.0f), x );
//}

inline
__m128 sign( const __m128 x )
{
	const __m128 zeros = _mm_setzero_ps();
	
	const __m128 m0 = _mm_cmpgt_ps(x, zeros);
	const __m128 m1 = _mm_cmplt_ps(x, zeros);
	__m128 ret = _mm_blendv_ps( zeros, _mm_set1_ps( 1.0f), m0 );
	return _mm_blendv_ps( ret, _mm_set1_ps(-1.0f), m1 );
}

inline
__m256 sign( const __m256 x )
{
	const __m256 zeros = _mm256_setzero_ps();
	const __m256 m0 = _mm256_cmp_ps( x, zeros, _CMP_GT_OQ );
	const __m256 m1 = _mm256_cmp_ps( x, zeros, _CMP_LT_OQ );
	__m256 ret = _mm256_blendv_ps( zeros, _mm256_set1_ps(1.0f), m0 );
	return _mm256_blendv_ps( ret, _mm256_set1_ps(-1.0f), m1 );
}

inline
__m512 sign( const __m512 x )
{
	const __m512 zeros = _mm512_setzero_ps();

	const __mmask16 m0 = _mm512_cmp_ps_mask( x, zeros, _CMP_GT_OQ );
	const __mmask16 m1 = _mm512_cmp_ps_mask( x, zeros, _CMP_LT_OQ );
	__m512 ret = _mm512_mask_blend_ps( m0, zeros, _mm512_set1_ps(  1.0f) );
	return _mm512_mask_blend_ps( m1, ret, _mm512_set1_ps( -1.0f) );	
}


///////////////////////////////////////////////////////////////////////////////


#include <vector>
#include <string>
#include <sstream>

namespace lpt {

	template<typename T> inline
	std::string R_to_string( const T &_input ) {
		std::stringstream ost;
		ost << _input;
		return std::string( ost.str() );
	}

	inline
	std::string to_string( const std::string &in_str ) {
		return in_str;
	}

	inline
	std::string to_string( const bool &_input ) {
		return _input?std::string("true"):std::string("false");
	}

	//inline
	//std::string to_string( const GLboolean &_input ) {
	//	return _input?std::string("true"):std::string("false");
	//}

	inline
	std::string to_string( const float &in_float ) {
		std::string tmp = R_to_string( in_float );
		size_t dotloc = tmp.find( '.' );

		if(dotloc != std::string::npos) {
			const int num_dec = 2;
			if( tmp.size() > dotloc+num_dec )
				tmp.erase( dotloc+num_dec+1, tmp.size()-1 );
			else
				while( tmp.size() < dotloc+num_dec+1)
					tmp.push_back( '0' );
		}

		return tmp;
	}

	template<typename T> inline
	std::string to_string( const T &input_var ) {
		return R_to_string( input_var );
	}

	//std::string time_to_string( const int64_t &time_ms );

	//////////////////////////////////////////////////////////////////////////

	//TODO: do streamversion of this
	template<typename T> inline
	void bitprint( const T in_val ) {
		for(int i=sizeof(T)*8-1; i+1; --i) {
			if(in_val&(1<<i)) std::cerr << 1;
			else std::cerr << 0;
		}
		return;
	}

	//template<typename T> inline
	//void hexprint( const T in_val ) {
	//	std::cerr << "0x" << std::hex << in_val; // << std::dec;
	//	//std::cerr.hex = false;
	//}


	inline
	std::vector<std::string> split( const std::string &in_str, const std::string &delim ) {
		std::vector<std::string> ret;
		std::string curstr = "";
		for(unsigned int i=0;i<in_str.length();++i) {
			bool match_found=false;
			for( unsigned int j=0;j<delim.length();++j) {
				if( in_str[i] == delim[j] ) {
					ret.push_back(curstr);
					curstr="";
					match_found=true;
					break;
				}
			}
			if(!match_found)
				curstr.push_back(in_str[i]);
		}

		ret.push_back(curstr);

		return ret;
	}
}

///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////

LARGE_INTEGER currentTime;
LARGE_INTEGER m_timer_frequency;
void init_timers()
{
	QueryPerformanceFrequency( &m_timer_frequency );
}
uint64_t gettime_ms()
{
	QueryPerformanceCounter(&currentTime);
	return static_cast<uint64_t>( static_cast<double>(currentTime.QuadPart) / static_cast<double>(m_timer_frequency.QuadPart) * 1000.0 );
}

///////////////////////////////////////////////////////////////////////////////

struct header_t
{ 
		int32_t dim_x, dim_y, dim_z;
		float32_t bb_mn_x, bb_mn_y, bb_mn_z;
		float32_t bb_mx_x, bb_mx_y, bb_mx_z;
};
struct sdf_t
{
	header_t header;
	float32_t *data = nullptr;

	#ifndef NDEBUG
	vec3_t *eval_points;
	#endif
};

struct aabb_t
{
	vec3_t mn;
	vec3_t mx;
	aabb_t() : mn(FLT_MAX,FLT_MAX,FLT_MAX), mx(-FLT_MAX,-FLT_MAX,-FLT_MAX) {}
};

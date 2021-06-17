#ifndef SVG_H__
#define SVG_H__

#include <stdlib.h>
#include <stdio.h>
#include <corecrt_memory.h>
#include <cfloat>
#include <cassert>

///////////////////////////////////////////////////////////
//TODO: transparencies: https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Fills_and_Strokes

namespace svg
{
	struct svg_t
	{
		FILE *fp;
		float width, height;
	};

	bool isvalid( svg_t &context )
	{
		return context.fp != NULL;
	}

	void html_header( svg_t &context )
	{
		assert( isvalid(context) );
		//TODO: input bgcolor
		char const * const footer = "<html><body bgcolor=\"#000000\"><div style=\"text-align:center\">\n";
		fputs(footer, context.fp);
	}

	void svg_header( svg_t &context )
	{
		assert( isvalid(context) );
		fprintf_s(context.fp,
				  "<svg width=\"%f\" height=\"%f\">\n",
				  context.width, context.height
		);
	}
	void html_footer( svg_t &context )
	{
		char const * const footer = "</body></html>\n";
		fputs(footer, context.fp);
	}
	void svg_footer( svg_t &context )
	{
		assert( isvalid(context) );
		char const * const footer = "</svg>\n<br>\n";
		fputs(footer, context.fp);
	}

	void open( svg_t &context, char const * const in_filename, float width, float height )
	{
		fopen_s(&context.fp, in_filename, "w+");
		assert( context.fp != NULL );

		context.width = width;
		context.height = height;

		html_header( context );
		svg_header(context);
	}

	void close( svg_t &context )
	{
		assert( isvalid(context) );
		svg_footer( context );
		html_footer( context );
		fclose(context.fp);
	}
	void begin_new_svg( svg_t &context )
	{
		svg_footer( context );
		svg_header( context );
	}

	void color__int_to_bytes( const unsigned int c, unsigned char &c_r, unsigned char &c_g, unsigned char &c_b )
	{
		c_r = (c>>16)&0xffu;
		c_g = (c>>8)&0xffu;
		c_b = (c>>0)&0xffu;
	}
	void color__bytes_to_int( const unsigned char c_r, const unsigned char c_g, const unsigned char c_b, unsigned int &c )
	{
		c = (c_r<<16) | (c_g<<8) | c_b;
	}

	// ====
	void drawline( svg_t &context, float p0_x, float p0_y, float p1_x, float p1_y, float strokewidth=1, unsigned int strokecolor=0xffffff )
	{
		p0_y = context.height - p0_y;
		p1_y = context.height - p1_y;

		unsigned char strokecolor_r, strokecolor_g, strokecolor_b; color__int_to_bytes( strokecolor, strokecolor_r, strokecolor_g, strokecolor_b );

		fprintf_s(context.fp,
				  "<line x1=\"%f\" y1=\"%f\" x2=\"%f\" y2=\"%f\" style=\"stroke:rgb(%d,%d,%d);stroke-width:%f\"/>\n",
				  p0_x, p0_y, p1_x, p1_y,
 				  strokecolor_r, strokecolor_g, strokecolor_b,
				  strokewidth
		);
	}

	//void drawlinestrip( svg_t &context, float const * const pts_x, float const * const pts_y, const int num_points, float strokewidth=1, unsigned int strokecolor=0xffffff )
	//{
	//	unsigned char strokecolor_r, strokecolor_g, strokecolor_b; color__int_to_bytes( strokecolor, strokecolor_r, strokecolor_g, strokecolor_b );
	//
	//	//  <polyline points="0,40 40,40 40,80 80,80 80,120 120,120 120,160" style="fill:white;stroke:red;stroke-width:4" />
	//
	//	fprintf_s(context.fp,
	//			  "<polyline points=\"");
	//
	//	for ( int i=0,n=num_points; i<n; ++i )
	//	{
	//		fprintf_s(context.fp,
	//				  "%f,%f ",
	//				  pts_x[i], pts_y[i]
	//		);
	//	}
	//
	//	fprintf_s(context.fp,
	//				  "\" style=\"stroke:rgb(%d,%d,%d);stroke-width:%f\"/>\n",
 	//				  strokecolor_r, strokecolor_g, strokecolor_b,
	//				  strokewidth
	//	);
	//}


	// ====
	void drawcircle( svg_t &context, float x, float y, float r, unsigned int fillcolor=0xffffff, float strokewidth=0, unsigned int strokecolor=0x000000)
	{
		y = context.height - y;

		unsigned char fillcolor_r, fillcolor_g, fillcolor_b;       color__int_to_bytes( fillcolor, fillcolor_r, fillcolor_g, fillcolor_b );
		unsigned char strokecolor_r, strokecolor_g, strokecolor_b; color__int_to_bytes( strokecolor, strokecolor_r, strokecolor_g, strokecolor_b );

		fprintf_s(context.fp,
			"<circle cx=\"%f\" cy=\"%f\" r=\"%f\" style=\"fill:rgb(%d,%d,%d);stroke-width:%f;stroke:rgb(%d,%d,%d)\"/>\n",
			x, y, r,
			fillcolor_r, fillcolor_g, fillcolor_b,
			strokewidth,
			strokecolor_r, strokecolor_g, strokecolor_b
		);
	}

	// ====
	void drawrect( svg_t &context, float x, float y, float w, float h, unsigned int fillcolor=0xffffff, float strokewidth=0, unsigned int strokecolor=0x000000 )
	{
		y = context.height - h - y;

		unsigned char fillcolor_r, fillcolor_g, fillcolor_b;       color__int_to_bytes( fillcolor, fillcolor_r, fillcolor_g, fillcolor_b );
		unsigned char strokecolor_r, strokecolor_g, strokecolor_b; color__int_to_bytes( strokecolor, strokecolor_r, strokecolor_g, strokecolor_b );
		
		fprintf_s(context.fp,
				  "<rect x=\"%f\" y=\"%f\" width=\"%f\" height=\"%f\" style=\"fill:rgb(%d,%d,%d);stroke-width:%f;stroke:rgb(%d,%d,%d)\"/>\n",
				  x, y, w, h,
				  fillcolor_r, fillcolor_g, fillcolor_b,
				  strokewidth,	  
				  strokecolor_r, strokecolor_g, strokecolor_b
		);
	}

	void drawtext( svg_t &context, float x, float y, char const * const str, unsigned int fillcolor )
	{
		y = context.height - y;

		unsigned char fillcolor_r, fillcolor_g, fillcolor_b; color__int_to_bytes( fillcolor, fillcolor_r, fillcolor_g, fillcolor_b );
		fprintf_s(context.fp,
				  "<text x=\"%f\" y=\"%f\" style=\"fill:rgb(%d,%d,%d)\">%s</text>\n", x, y, fillcolor_r, fillcolor_g, fillcolor_b, str );
	}

	void drawtext_ctr( svg_t &context, float x, float y, char const * const str, unsigned int fillcolor=0x808080u )
	{
		y = context.height - y;

		unsigned char fillcolor_r, fillcolor_g, fillcolor_b; color__int_to_bytes( fillcolor, fillcolor_r, fillcolor_g, fillcolor_b );

		fprintf_s(context.fp,
			"<text x=\"%f\" y=\"%f\" text-anchor=\"middle\" style=\"fill:rgb(%d,%d,%d)\">%s</text>\n", x, y, fillcolor_r, fillcolor_g, fillcolor_b, str );
	}
	
	void drawtext_ctr_vert( svg_t &context, float x, float y, char const * const str, unsigned int fillcolor=0x808080u )
	{
		y = context.height - y;

		unsigned char fillcolor_r, fillcolor_g, fillcolor_b; color__int_to_bytes( fillcolor, fillcolor_r, fillcolor_g, fillcolor_b );

		fprintf_s(context.fp,
			"<text x=\"%f\" y=\"%f\" text-anchor=\"middle\" transform=\"rotate(-90 %f,%f)\" style=\"fill:rgb(%d,%d,%d)\">%s</text>\n", x, y, x, y, fillcolor_r, fillcolor_g, fillcolor_b, str );
	}

	void comment( svg_t &context, char const * const str )
	{
		fprintf_s(context.fp,
			"<!-- %s -->\n", str );
	}
} //namespace svg

namespace svg_utils
{
	void calc_series_minmax( float const * const pt_x, float const * const pt_y, const int num_points, float &min_x, float &max_x, float &min_y, float &max_y )
	{
		min_x = FLT_MAX;
		max_x = -FLT_MAX;
		min_y = FLT_MAX;
		max_y = -FLT_MAX;
		for ( int i=0,n=num_points; i<n; ++i )
		{
			float x = pt_x[i];
			min_x = (x<min_x) ? x : min_x;
			max_x = (x>max_x) ? x : max_x;
			
			float y = pt_y[i];
			min_y = (y<min_y) ? y : min_y;
			max_y = (y>max_y) ? y : max_y;
		}
	}
	void union_minmax( float min_x0, float max_x0, float min_y0, float max_y0,
					   float min_x1, float max_x1, float min_y1, float max_y1,
					   float &min_x, float &max_x, float &min_y, float &max_y )
	{
		assert( min_x0 <= max_x0 );
		assert( min_x1 <= max_x1 );
		assert( min_y0 <= max_y0 );
		assert( min_y1 <= max_y1 );

		min_x = (min_x0 < min_x1 ) ? min_x0 : min_x1;
		max_x = (max_x0 > max_x1 ) ? max_x0 : max_x1;

		min_y = (min_y0 < min_y1 ) ? min_y0 : min_y1;
		max_y = (max_y0 > max_y1 ) ? max_y0 : max_y1;

		assert( min_x <= max_x );
		assert( min_y <= max_y );
	}

	void calc_graph_bounds( float x, float y, float w, float h,
						    float &bb_x, float &bb_y, float &bb_w, float &bb_h )
	{
		float ofs_x = 0.1f*w;
		float ofs_y = 0.1f*h;
		float ofs = (ofs_x < ofs_y ) ? ofs_x : ofs_y;
		bb_x = x + ofs;
		bb_y = y + ofs;
		bb_w = w - 2.0f * ofs;
		bb_h = h - 2.0f * ofs;
	}

	void drawgraph_series( svg::svg_t &context,
						   float graph_x, float graph_y, float graph_w, float graph_h,
						   float min_x, float max_x, float min_y, float max_y,
						   float const * const pt_x, float const * const pt_y, const int num_points,
						   unsigned int color=0xff0000 )
	{
		svg::comment( context, "begin-series");
		float bb_x, bb_y, bb_w, bb_h; 
		calc_graph_bounds( graph_x, graph_y, graph_w, graph_h, bb_x, bb_y, bb_w, bb_h );

		for ( int i=0,n=num_points; i<n; ++i )
		{
			float p0_x = pt_x[i+0];
			float p0_y = pt_y[i+0];

			float p0_x_nm = (p0_x-min_x) / (max_x-min_x);
			float p0_y_nm = (p0_y-min_y) / (max_y-min_y);

			float p0_x_g = bb_x + p0_x_nm * bb_w;
			float p0_y_g = bb_y + p0_y_nm * bb_h;

			svg::drawcircle(context, p0_x_g, p0_y_g, 3.0, color );

			if ( i < num_points-1)
			{

				float p1_x = pt_x[i+1];
				float p1_y = pt_y[i+1];

				float p1_x_nm = (p1_x-min_x) / (max_x-min_x);
				float p1_y_nm = (p1_y-min_y) / (max_y-min_y);

				float p1_x_g = bb_x + p1_x_nm * bb_w;
				float p1_y_g = bb_y + p1_y_nm * bb_h;

				drawline( context,
							p0_x_g, p0_y_g,
							p1_x_g, p1_y_g,
							1, color );
			}
		}
		svg::comment( context, "end-series");
	}

	void drawgraph( svg::svg_t &context, float x, float y, float w, float h )
	{
		drawrect( context, x, y, w, h, 0xffffffu, 0 );

		float bb_x, bb_y, bb_w, bb_h; 
		calc_graph_bounds( x, y, w, h, bb_x, bb_y, bb_w, bb_h );

		drawrect( context, bb_x, bb_y, bb_w, bb_h, 0xffffffu, 1, 0x000000u );

		//TODO: grid (at major lines? or divided)
	}

	//TODO: this ia a bit manual, should be able to align to major-lines in data...
	void drawgraph_grid( svg::svg_t &context,
						 float graph_x, float graph_y, float graph_w, float graph_h,
						 int major_subd_x, int major_subd_y,
						 unsigned int major_color = 0x404040u
						 /*int minor_subd_x, int minor_subd_y,
						 unsigned int minor_color = 0x808080u*/ )
	{
		svg::comment( context, "begin-grid");
		
		float bb_x, bb_y, bb_w, bb_h; 
		calc_graph_bounds( graph_x, graph_y, graph_w, graph_h, bb_x, bb_y, bb_w, bb_h );

		for ( int i=0,n=major_subd_x-1; i<n; ++i )
		{
			float t = static_cast<float>(i) / static_cast<float>(n);

			float lx = bb_x + t * bb_w;
			svg::drawline( context, lx, bb_y, lx, bb_y+bb_h, 1, major_color );
		}

		for ( int i=0,n=major_subd_y-1; i<n; ++i )
		{
			float t = static_cast<float>(i) / static_cast<float>(n);

			float ly = bb_y + t * bb_h;
			svg::drawline( context, bb_x, ly, bb_x+bb_w, ly, 1, major_color );
		}

		svg::comment( context, "end-grid");
	}

	void drawgraph_axislabels( svg::svg_t &context,
		float graph_x, float graph_y, float graph_w, float graph_h,
		char const * const label_x, char const * const label_y )
	{
		float bb_x, bb_y, bb_w, bb_h; 
		calc_graph_bounds( graph_x, graph_y, graph_w, graph_h, bb_x, bb_y, bb_w, bb_h );

		svg::drawtext_ctr( context, graph_x + 0.5f*graph_w, 0.5f*(bb_y+graph_y), label_x );

		svg::drawtext_ctr_vert( context, 0.5f*(bb_x+graph_x), 0.5f*graph_h, label_y );
	}
} //namespace svg_utils

////////////////////////////////////


//void dump_debug_html( char const * const path, int in_width, int in_height, vec2_t *pts, int32_t num_points, bbox_t bb )
void test_html_dump()
{
	svg::svg_t context;
	svg::open(context, "dummy_output.html", 512, 512);

	svg::drawrect( context, 0, 0, 512, 512, 0x808080 );
	svg::drawrect( context, 25, 25, 50,50, 0x404040);

	svg::drawrect( context, 100,100, 150,150, 0xaabbcc);
	
	{
		unsigned int c;
		svg::color__bytes_to_int(255,0,0, c); //note: for when you reaaaally want to use decimal...
		svg::drawcircle( context, 50, 50, 25, 0xffaaaa, 1, c );
	}

	svg::drawline( context, 25,25, 75,25, 1, 0xff0000 );
	svg::drawline( context, 75,25, 75,75, 2, 0x00ff00 );
	svg::drawline( context, 75,75, 25,75, 3, 0x0000ff );
	svg::drawline( context, 25,75, 25,25, 4, 0xff00ff );

	{
		svg::comment(context, "begin-graph" );
		svg_utils::drawgraph( context, 200,200, 200,100 );

		enum { NUM_GRAPH_PTS0 = 8 };
		float pts0_x[NUM_GRAPH_PTS0] = {1, 2, 3, 4, 5, 6, 7, 8};
		float pts0_y[NUM_GRAPH_PTS0] = {1, 2, 1, 2, 1, 2, 1, 3};

		enum { NUM_GRAPH_PTS1 = 5 };
		float pts1_x[NUM_GRAPH_PTS1] = {0.5, 1.5, 2.5, 3.5, 4.5};
		float pts1_y[NUM_GRAPH_PTS1] = {2, 3, 2, 3, 1};

		float min_x0, max_x0, min_y0, max_y0;
		svg_utils::calc_series_minmax(pts0_x, pts0_y, NUM_GRAPH_PTS0, min_x0, max_x0, min_y0, max_y0 );

		float min_x1, max_x1, min_y1, max_y1;
		svg_utils::calc_series_minmax(pts1_x, pts1_y, NUM_GRAPH_PTS1, min_x1, max_x1, min_y1, max_y1 );

		float min_x, max_x, min_y, max_y;
		svg_utils::union_minmax( min_x0, max_x0, min_y0, max_y0,
						   min_x1, max_x1, min_y1, max_y1,
						   min_x, max_x, min_y, max_y );

		svg_utils::drawgraph_series( context, 200,200, 200,100, min_x, max_x, min_y, max_y, pts0_x, pts0_y, NUM_GRAPH_PTS0, 0xff0000 );
		svg_utils::drawgraph_series( context, 200,200, 200,100, min_x, max_x, min_y, max_y, pts1_x, pts1_y, NUM_GRAPH_PTS1, 0x00ff00 );

		svg::comment(context, "end-graph" );
	}
	
	svg::close(context);
}

#endif //SVG_H__

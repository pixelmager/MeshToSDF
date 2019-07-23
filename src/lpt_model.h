#pragma once 

#include <iostream>
#include <vector>

#include <assert.h>

// assimp include files. These three are usually needed.
#include <assimp/cimport.h>         // C importer interface
//#include <assimp/Importer.hpp>    // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags
#include <assimp/vector3.h>
#include <assimp/matrix4x4.h>

#include <sdf_support.h>

namespace lpt
{
	struct indexed_triangle_mesh_t
	{
		std::vector<uint32_t> tri_indices; //3 x num_triangles
		std::vector<float32_t> positions;
	};


	//note: file started as Sample_SimpleOpenGL.c from assimp 3.0-1270 SDK

	struct float3_t
	{
		float32_t x, y, z;
		//float3(){}
		//float3( float32_t in_x, float32_t in_y, float32_t in_z ) : x(in_x), y(in_y), z(in_z) {}
	};

	struct float4_t
	{
		float32_t x, y, z, w;

		float4_t(){}
		float4_t( float32_t in_x, float32_t in_y, float32_t in_z, float32_t in_w ) : x(in_x), y(in_y), z(in_z), w(in_w) {}
	};
	struct float4x4_t
	{
		float32_t m_mat[16];
	};

	float4_t operator-(const float4_t &v)
	{
		float4_t ret;
		ret.x = -v.x;
		ret.y = -v.y;
		ret.z = -v.z;
		ret.w = -v.w;
		return ret;
	}
	float4_t operator*( const float4_t &v, const float32_t c )
	{
		float4_t ret;
		ret.x = v.x * c;
		ret.y = v.y * c;
		ret.z = v.z * c;
		ret.w = v.w * c;
		return ret;
	}
	float4_t operator*(const float32_t c, const float4_t &v)
	{
		return operator*(v,c);
	}
	float4_t operator*(const float4_t &v0, const float4_t &v1)
	{
		float4_t ret;
		ret.x = v0.x * v1.x;
		ret.y = v0.y * v1.y;
		ret.z = v0.z * v1.z;
		ret.w = v0.w * v1.w;
		return ret;
	}
	float4_t operator+(const float4_t &v0, const float4_t &v1)
	{
		float4_t ret;
		ret.x = v0.x + v1.x;
		ret.y = v0.y + v1.y;
		ret.z = v0.z + v1.z;
		ret.w = v0.w + v1.w;
		return ret;
	}


	typedef float4_t quaternion_t;

	//////////////////////////////////////////////////////
	typedef int32_t idxSurf;
	typedef int32_t idxSurfTransform;
	typedef int32_t idxModel;
	typedef int32_t idxGeom;
	typedef int32_t idxAnim;
	//////////////////////////////////////////////////////

	typedef enum { VD_POSITIONS=0, VD_NORMALS, VD_TANGENT, VD_BITANGENT, VD_UV0, VD_UV1, VD_NUM_TYPES } eVertexDataType;
	static const int32_t vattrib_bits[ VD_NUM_TYPES ] =  { 1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5 };
	static const int8_t vattrib_dim_per_elem[VD_NUM_TYPES] = { 3, 3, 3, 3, 2, 2 };

	//note: always floats
	//vertex-stream? vertex-... naaaming!
	struct vertexData_t
	{
		std::vector<float32_t>	data;
		eVertexDataType			datatype;
	};

	//note: indexed triangles
	struct surface_t
	{
		std::vector<uint32_t>		indices;
		vertexData_t				posdata;    //note: float3, separate to speed up depth-passes
		std::vector<vertexData_t>	vertexdata; //TODO: additional vertex-data
		//TODO: bbox_t oobb;
		int32_t						mat_idx; //TODO: into nothing...
		//TODO: uint32_t					used_vertex_attribs; //pos, norm, uv0, uv1 ... uvn?
	};

	struct model_t
	{
		std::string						name; //TODO: debug only? move to scenedata?
		std::vector<idxSurf>			surfindices; //note: array of surfs, may be empty

		//TODO: extract parent-idx to scene...
		//note: scene-hierarchy... move to scene?
		//idxModel						parent_idx;  //note: may be null
		//std::vector<idxModel>			children_indices; //note: may be empty
	};

	struct anim_t
	{
		std::string					name;
		std::vector<float32_t>		animkeys_time; // [0;1] ...TODO: save as int32_t instead
		std::vector<float3_t>			animkeys_pos;
		std::vector<float3_t>			animkeys_scl;
		std::vector<quaternion_t>	animkeys_rot;
	};

	//TODO: camera needs to be contained as well?
	struct sceneData_t
	{
		std::vector<model_t>		models;				//note: flat array of scene-nodes
		std::vector<float4x4_t>		models_transforms;	//note: array model_to_parent transforms, same size as models
		std::vector<idxAnim>		models_anims;		//note: array model_to_parent transforms, same size as models
		std::vector<idxModel>		models_paridx;		//note: parent index

		std::vector<surface_t>		surfs;				//note: array of surfs, models have indices into this
		std::vector<anim_t>			anims;				//note: models have indices into this

		//TODO: move all scene-vertex-data to single array?
	};

	float32_t lcl_min( float32_t a, float32_t b )
	{
		return (a<b) ? a : b;
	}
	float32_t lcl_max( float32_t a, float32_t b )
	{
		return (a>b) ? a : b;
	}

	// ====
	void get_bounding_box_for_node ( aiScene const * const scene,
									 aiNode const * const nd, 
									 float3_t &mn, 
									 float3_t &mx, 
									 aiMatrix4x4* trans )
	{
		aiMatrix4x4 prev = *trans;
		aiMultiplyMatrix4( trans, &nd->mTransformation );

		for ( unsigned int i=0, nm=nd->mNumMeshes; i<nm; ++i )
		{
			aiMesh const * const mesh = scene->mMeshes[ nd->mMeshes[i] ];
			for ( unsigned int t=0, nv=mesh->mNumVertices; t<nv; ++t)
			{

				aiVector3D tmp = mesh->mVertices[t];
				aiTransformVecByMatrix4(&tmp,trans);

				mn.x = lcl_min(mn.x, tmp.x);
				mn.y = lcl_min(mn.y, tmp.y);
				mn.z = lcl_min(mn.z, tmp.z);

				mx.x = lcl_max(mx.x, tmp.x);
				mx.y = lcl_max(mx.y, tmp.y);
				mx.z = lcl_max(mx.z, tmp.z);
			}
		}

		for ( unsigned int i=0, ic=nd->mNumChildren; i<ic; ++i )
		{
			get_bounding_box_for_node( scene, nd->mChildren[i], mn, mx, trans );
		}

		*trans = prev;
	}

	// ====
	void get_bounding_box ( aiScene const * const scene, float3_t &min, float3_t &max)
	{
		aiMatrix4x4 trans;
		aiIdentityMatrix4(&trans);

		min.x = min.y = min.z =  1e10f;
		max.x = max.y = max.z = -1e10f;
		get_bounding_box_for_node( scene, scene->mRootNode, min, max, &trans );
	}


	// ====
	aiScene const * loadasset ( char const * const path, float3_t &scene_center, float3_t &scene_min, float3_t &scene_max )
	{
		// we are taking one of the postprocessing presets to avoid
		// spelling out 20+ single postprocessing flags here.
		aiScene const * const scene = aiImportFile( path, aiProcessPreset_TargetRealtime_MaxQuality );

		if ( scene != nullptr )
		{
			get_bounding_box(scene, scene_min, scene_max);
			scene_center.x = (scene_min.x + scene_max.x) * 0.5f;
			scene_center.y = (scene_min.y + scene_max.y) * 0.5f;
			scene_center.z = (scene_min.z + scene_max.z) * 0.5f;
		}

		return scene;
	}

	// ====
	void copyTransform( const aiMatrix4x4 in_mat, float4x4_t &out_mat )
	{
		// update transform for gl
		aiMatrix4x4 glmat = in_mat;
		aiTransposeMatrix4(&glmat);
		memcpy( &out_mat, &glmat, 16 * sizeof(float) );
	}

	// ====
	void processNodeGeometry( aiScene const * const scene,
							  aiNode const * const nd,
							  model_t &newmodel,
							  sceneData_t * const out_scene )
	{
		const uint32_t num_meshes = nd->mNumMeshes;
		for ( unsigned int midx=0; midx < num_meshes; ++midx )
		{
			const uint32_t aimesh_idx = nd->mMeshes[ midx ];
			aiMesh const * const aimesh = scene->mMeshes[ aimesh_idx ];

			//TODO: wtfs this? vertex-deformation?
			//LPT_WARNING( aimesh->mNumAnimMeshes==0, (std::string("# warn: scene has ") + to_string(aimesh->mNumAnimMeshes) + std::string(" animMeshes, but not read...")).c_str() );

			//std::cout << "\tmesh-name \"" << aimesh->mName.C_Str() << "\"" << std::endl;
			//std::cout << "\tmat-index: " << aimesh->mMaterialIndex << std::endl;

			//TODO: improve warning
			const bool istris = (aimesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE) != 0;
			//LPT_WARNING( istris, (std::string("ONLY tris are supported, skipping mesh \"") + std::string(aimesh->mName.C_Str()) + std::string("\" ") + lpt::to_string(aimesh->mPrimitiveTypes) ).c_str() );
			if ( aimesh->HasFaces() && istris )
			{
				const int32_t newsurf_idx = static_cast<int32_t>( out_scene->surfs.size() );
				out_scene->surfs.push_back( surface_t() );
				surface_t &newsurf = out_scene->surfs.back();
				newsurf.mat_idx = static_cast<int32_t>( aimesh->mMaterialIndex );

				newmodel.surfindices.push_back( newsurf_idx );

				assert( aimesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE && "ONLY tris are supported" );

				const uint32_t num_mesh_tris = aimesh->mNumFaces;

				newsurf.indices.resize( 3*num_mesh_tris ); //note: assumes tris
				for ( unsigned int t = 0; t < num_mesh_tris; ++t )
				{
					const aiFace &face = aimesh->mFaces[t];
					assert( face.mNumIndices == 3 && "only tris are supported" );
					newsurf.indices.at(3*t+0) = face.mIndices[0];
					newsurf.indices.at(3*t+1) = face.mIndices[1];
					newsurf.indices.at(3*t+2) = face.mIndices[2];
				}
				assert( (newsurf.indices.size() % 3) == 0 );
				assert( newsurf.indices.size() == aimesh->mNumFaces*3 );

				assert( aimesh->mNumVertices > 0 );
				assert( aimesh->mVertices != nullptr );


				assert( aimesh->HasPositions() );
				newsurf.posdata.datatype = eVertexDataType::VD_POSITIONS;
				newsurf.posdata.data.resize( 3*aimesh->mNumVertices );
				memcpy( &(newsurf.posdata.data.front()), aimesh->mVertices, 3 * aimesh->mNumVertices * sizeof( float ) );

				if ( aimesh->HasNormals() )
				{
					assert( aimesh->mNormals != nullptr );
					assert( aimesh->mNumVertices > 0 );

					newsurf.vertexdata.push_back( vertexData_t() );
					vertexData_t &normals = newsurf.vertexdata.back();
					normals.datatype = eVertexDataType::VD_NORMALS;
					normals.data.resize( 3 * aimesh->mNumVertices );
					memcpy( &(normals.data.front()), aimesh->mNormals, 3 * aimesh->mNumVertices * sizeof( float ) );
				}
				if ( aimesh->HasTangentsAndBitangents() )
				{
					assert( aimesh->mTangents != nullptr );
					assert( aimesh->mBitangents != nullptr );
					assert( aimesh->mNumVertices > 0 );

					//TODO
					//float *tans = new float[ 3 * aimesh->mNumVertices ]; //xyz
					//memcpy( tans, aimesh->mTangents, 3 * aimesh->mNumVertices * sizeof( float ) );
					//
					//float *bitan = new float[ 3 * mesh->mNumVertices ]; //xyz
					//memcpy( bitan, aimesh->mBitangents, 3 * aimesh->mNumVertices * sizeof( float ) );
				}

				//LPT_WARNING( aimesh->GetNumUVChannels() <= 2, "only 2 uv sets supported" );
				if ( aimesh->HasTextureCoords(0) )
				{
					assert( aimesh->mTextureCoords[0] != nullptr );
					//LPT_WARNING_PARANOID( aimesh->mNumUVComponents[0] == 2, "only 2 dimensional uvs are supported" /*TODO: found aimesh->mNumUVComponents[0]*/ );
					assert( aimesh->mNumVertices > 0 );

					newsurf.vertexdata.push_back( vertexData_t() );
					vertexData_t &uv0 = newsurf.vertexdata.back();
					uv0.datatype = eVertexDataType::VD_UV0;
					uv0.data.resize( 2 * aimesh->mNumVertices );

					aiVector3D const * const arr = aimesh->mTextureCoords[0];
					//note: manual copy of ST, skipping P... which appears to be in most models, and 0... :p
					for( int i=0, n=aimesh->mNumVertices; i<n; ++i )
					{
						uv0.data[2*i+0] = arr[i].x;
						uv0.data[2*i+1] = arr[i].y;	
					}
				}
				if ( aimesh->HasTextureCoords(1) )
				{
					assert( aimesh->mTextureCoords[1] != nullptr );
					//LPT_WARNING_PARANOID( aimesh->mNumUVComponents[1] == 2, "only 2 dimensional uvs are supported" /*TODO: found aimesh->mNumUVComponents[1]*/ );
					assert( aimesh->mNumVertices > 0 );

					//note: same as above, TODO: merge
					newsurf.vertexdata.push_back( vertexData_t() );
					vertexData_t &uv1 = newsurf.vertexdata.back();
					uv1.datatype = eVertexDataType::VD_UV1;
					uv1.data.resize( 2 * aimesh->mNumVertices );

					aiVector3D const * const arr = aimesh->mTextureCoords[0];
					//note: manual copy of ST, skipping P... which appears to be in most models, and 0... :p
					for( int i=0, n=aimesh->mNumVertices; i<n; ++i )
					{
						uv1.data[2*i+0] = arr[i].x;
						uv1.data[2*i+1] = arr[i].y;
					}
				}

				assert( aimesh->GetNumColorChannels() <= 1 && "only a single vertexcolor-channel supported" );
				if ( aimesh->HasVertexColors(0) )
				{
					assert( aimesh->mColors[0] != nullptr );

					//TODO
					std::cout << "# WARN: mesh \"" << aimesh->mName.C_Str() << "\" has vertex-colors, but not read..." << std::endl;
				}
			}

			if ( aimesh->HasBones() )
			{
				assert( aimesh->mBones );
				assert( aimesh->mNumBones > 0 );
				//TODO?
				std::cout << "# warn: \"" << aimesh->mName.C_Str() << "\" has bones, but not read..." << std::endl;
			}
		}
	}


	// ====
	void recursiveReadNode( aiScene const * const scene,
							aiNode const * const nd,
							sceneData_t * const out_scene,
							const int32_t parent_idx )
	{
		//std::cout << "node \"" << nd->mName.C_Str() << "\"" << " meshes: " << nd->mNumMeshes << " children: " << nd->mNumChildren << std::endl;

		const int32_t newmodel_idx = static_cast<int32_t>( out_scene->models.size() );
		out_scene->models.push_back( model_t() );
		model_t &newmodel = out_scene->models.at( newmodel_idx );
		newmodel.name = nd->mName.C_Str();
		out_scene->models_paridx.push_back( parent_idx );
		out_scene->models_anims.push_back( -1 );

		//if ( parent_idx > 0 )
		//	out_scene->models.at( parent_idx ).children_indices.push_back( newmodel_idx );

		out_scene->models_transforms.push_back( float4x4_t() );
		float4x4_t &newmodel_tf = out_scene->models_transforms[newmodel_idx];
		copyTransform( nd->mTransformation, newmodel_tf );

		//TODO: assimp support for instancing?

		processNodeGeometry( scene, nd, newmodel, out_scene );

		//note: traverse children
		for ( unsigned int n = 0; n < nd->mNumChildren; ++n )
			recursiveReadNode( scene, nd->mChildren[n], out_scene, newmodel_idx );
	}

	// ====
	sceneData_t* readAssImpScene( aiScene const * const scene )
	{
		sceneData_t *out_scenedata = new sceneData_t();

		//std::cout << "# reading scene-nodes..." << std::endl;
		recursiveReadNode( scene, scene->mRootNode, out_scenedata, -1 );

		//readAnimations( scene, out_scenedata );

		//readMaterials( scene );

		return out_scenedata;
	}

	// ====
	bool file_exists( const std::string &in_fullpath )
	{
		struct __stat64 stFileInfo;
		int statres = _stat64( in_fullpath.c_str(), &stFileInfo);
		if(statres == 0) {
			return true;
		} else {
			//TODO: lookup return value for proper error-reporting
			return false;
		}
	}

	// ====
	sceneData_t* assimp_loadmodel( const std::string &model_path )
	{
		assert( lpt::file_exists( model_path ) );

		// get a handle to the predefined STDOUT log stream and attach
		// it to the logging system. It remains active for all further
		// calls to aiImportFile(Ex) and aiApplyPostProcessing.
		aiLogStream stream = aiGetPredefinedLogStream( aiDefaultLogStream_STDOUT, NULL );
		aiAttachLogStream(&stream);

		// ... same procedure, but this stream now writes the
		// log messages to assimp_log.txt
		stream = aiGetPredefinedLogStream( aiDefaultLogStream_FILE, "bin/assimp_log.txt");
		aiAttachLogStream(&stream);

		// the model name can be specified on the command line. If none
		// is specified, we try to locate one of the more expressive test
		// models from the repository (/models-nonbsd may be missing in
		// some distributions so we need a fallback from /models!).
		float3_t scene_ctr, scene_min, scene_max;
		aiScene const * const scene = loadasset( model_path.c_str(), scene_ctr, scene_min, scene_max );
		if ( scene == nullptr )
		{
			std::cout << "## ERRR: couldn't find model \"" << model_path << "\"" << std::endl;
			exit(-1);
		}
		else
		{
			std::cout << "# read file " << model_path << std::endl;
		}
		//#if !defined( LPT_FINAL_RELEASE )
		std::cout << "scene center: (" << scene_ctr.x << ", " << scene_ctr.y << ", " << scene_ctr.z << " )" << std::endl;
		std::cout << "scene bbox-min: (" << scene_min.x << ", " << scene_min.y << ", " << scene_min.z << " )" << std::endl;
		std::cout << "scene bbox-max: (" << scene_max.x << ", " << scene_max.y << ", " << scene_max.z << " )" << std::endl;
		//#endif //LPT_FINAL_RELEASE

		sceneData_t *scenedata = readAssImpScene( scene );
	
		// cleanup - calling 'aiReleaseImport' is important, as the library 
		// keeps internal resources until the scene is freed again. Not 
		// doing so can cause severe resource leaking.
		aiReleaseImport(scene);

		// We added a log stream to the library, it's our job to disable it
		// again. This will definitely release the last resources allocated
		// by Assimp.
		aiDetachAllLogStreams();

		return scenedata;
	}

	void append_surface( surface_t &out_surf, const surface_t &surf )
	{
		out_surf.indices.reserve( out_surf.indices.size() + surf.indices.size() );
		out_surf.posdata.data.reserve( out_surf.posdata.data.size() + surf.posdata.data.size() );

		assert( out_surf.posdata.data.size() % 3 == 0 );
		const uint32_t baseidx = static_cast<uint32_t>( out_surf.posdata.data.size() / 3 );

		for ( size_t i=0, n=surf.indices.size(); i<n; ++i )
		{
			uint32_t newidx = baseidx + surf.indices[i];
			assert( 3*newidx < out_surf.posdata.data.size() + surf.posdata.data.size() );
			out_surf.indices.push_back( newidx );
		}
		
		for ( size_t i=0,n=surf.posdata.data.size(); i<n; ++i )
		{
			out_surf.posdata.data.push_back( surf.posdata.data[i] );
		}

		#ifndef NDEBUG
		const size_t num_pos = out_surf.posdata.data.size();
		for ( size_t i = 0, n=out_surf.indices.size(); i<n; ++i )
		{
			const uint32_t idx = out_surf.indices[i];
			assert( 3*idx < num_pos );
		}
		#endif

		//TODO: vertexdata
	}

	void bake_to_single_surf( sceneData_t const * const scene, surface_t &out_surf )
	{
		out_surf.indices.clear();
		out_surf.posdata.datatype = eVertexDataType::VD_POSITIONS;
		out_surf.posdata.data.clear();
		out_surf.vertexdata.clear(); //note: skip
		out_surf.mat_idx = -1; //note: skip

		for ( size_t modelidx=0, num_models=scene->models.size(); modelidx < num_models; ++modelidx )
		{
			const model_t &mdl = scene->models[ modelidx ];

			for ( size_t surfidx=0, num_surfs=mdl.surfindices.size(); surfidx < num_surfs; ++surfidx )
			{
				const surface_t &surf = scene->surfs[ surfidx ];
				append_surface( out_surf, surf );
			}
		}
	}

	indexed_triangle_mesh_t* loadmodel_assimp__posonly( const std::string &model_path )
	{
		sceneData_t *scene = assimp_loadmodel( model_path );

		surface_t surf;
		bake_to_single_surf( scene, surf );

		indexed_triangle_mesh_t *ret = new indexed_triangle_mesh_t();
		ret->tri_indices = surf.indices;
		ret->positions = surf.posdata.data;
		
		//assert( surf.posdata.data.size() % 3 == 0 );
		//ret->positions.reserve( surf.posdata.data.size() / 3 );
		//		
		//for ( size_t i=0, n=surf.posdata.data.size() / 3; i<n; ++i )
		//{
		//	ret->positions.push_back( float3( surf.posdata.data[3*i+0],
		//									  surf.posdata.data[3*i+0],
		//									  surf.posdata.data[3*i+0] ) );
		//}

		return ret;
	}

} //namespace lpt

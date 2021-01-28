#pragma once

#define PROFILER_NONE 0
#define PROFILER_MICROPROFILE 1
#define PROFILER_SUPERLUMINAL 2
#define PROFILER_TRACY 3

//#define USE_PROFILER PROFILER_MICROPROFILE
//#define USE_PROFILER PROFILER_SUPERLUMINAL
//#define USE_PROFILER PROFILER_TRACY
#define USE_PROFILER PROFILER_NONE

//TODO: if (USE_PROFILER == PROFILER_MICROPROFILE)
// microprofile.cpp still compiles, inline in .h?
#pragma comment( lib, "Ws2_32" )
#pragma comment( lib, "winmm" )
//TODO: endif (USE_PROFILER == PROFILER_MICROPROFILE)

#if ( USE_PROFILER == PROFILER_MICROPROFILE )

//#define MICROPROFILE_ENABLED 1

#include <stdio.h>
#include <stdarg.h>
#include <string>
#include <thread>
#include <atomic>
//#include "unistd.h"

//define MICROPROFILE_WEBSERVER_MAXFRAMES 100
#include "microprofile\microprofile.h"

//#pragma comment( lib, "Ws2_32" )
//#pragma comment( lib, "winmm" )

enum { NUM_COLS=6};
uint32_t colors[NUM_COLS] = { 0xFF6188,	0xA9DC76, 0xFFD866, 0xFC9867, 0xAB9DF2, 0x78DCE8 };
int curcol = 0;

#define PROFILE_ENTER(M) MICROPROFILE_ENTERI("defaultgroup", (#M), colors[(curcol++)%NUM_COLS] )
#define PROFILE_LEAVE(M) MICROPROFILE_LEAVE()
#define PROFILE_SCOPE(M) MICROPROFILE_SCOPEI( "defaultgroup", (#M), colors[(curcol++)%NUM_COLS] )
#define PROFILE_FUNC() MICROPROFILE_SCOPEI( "defaultgroup", __FUNCTION__, colors[(curcol++)%NUM_COLS] )

#define PROFILE_THREADNAME(M) MicroProfileOnThreadCreate(M)
//TODO: onthreadexit?

void init_profiler()
{
	//MicroProfileOnThreadCreate("Main");
	MicroProfileSetEnableAllGroups(true);
	MicroProfileSetForceMetaCounters(true);
	MicroProfileStartContextSwitchTrace();
}

void deinit_profiler()
{
	printf( "\nmicroprofile outputting..." );
	MicroProfileDumpFileImmediately("sdf_profile.html", nullptr, nullptr);
	MicroProfileShutdown();
}

//MICROPROFILE_DECLARE_LOCAL_ATOMIC_COUNTER(ThreadsStarted);
//MICROPROFILE_DEFINE_LOCAL_ATOMIC_COUNTER(ThreadSpinSleep, "/runtime/spin_sleep");
//MICROPROFILE_DECLARE_LOCAL_COUNTER(LocalCounter);
//MICROPROFILE_DEFINE_LOCAL_COUNTER(LocalCounter, "/runtime/localcounter");

#elif ( USE_PROFILER == PROFILER_SUPERLUMINAL )

#pragma comment( lib, "PerformanceAPI_MD" )

#include <Superluminal/PerformanceAPI.h>

#define PERFORMANCEAPI_ENABLED 1

void init_profiler() {}
void deinit_profiler() {}
#define PROFILE_FUNC() PERFORMANCEAPI_INSTRUMENT_FUNCTION()
#define PROFILE_ENTER(M) PerformanceAPI::BeginEvent(#M)
#define PROFILE_LEAVE(M) PerformanceAPI::EndEvent()
#define PROFILE_SCOPE(M) PERFORMANCEAPI_INSTRUMENT(#M)
#define PROFILE_THREADNAME(M) PerformanceAPI::SetCurrentThreadName(#M)


#elif( USE_PROFILER == PROFILER_TRACY )

#pragma comment( lib, "DbgHelp.lib" )
#include "tracy/TracyC.h"
#include "tracy/Tracy.hpp"
#include "Tracy/TracyClient.cpp"

void init_profiler() {}
void deinit_profiler() {}
#define PROFILE_FUNC() ZoneScoped
//#define PROFILE_ENTER(M) TracyCZoneN( TracyConcat(__tracy, __LINE__) ,(M),true)
#define PROFILE_ENTER(M) TracyCZoneN( M, #M, true )
#define PROFILE_LEAVE(M) TracyCZoneEnd(M)
#define PROFILE_SCOPE(M) ZoneScopedN(#M)
#define PROFILE_THREADNAME(M) TracyCSetThreadName(M)

#else

void init_profiler() {}
void deinit_profiler() {}
#define PROFILE_FUNC()
#define PROFILE_ENTER(M)
#define PROFILE_LEAVE(M)
#define PROFILE_SCOPE(M)
#define PROFILE_THREADNAME(M)
//SetThreadDescription( GetCurrentProcess(), (M) );

#endif

#pragma once

#define PROFILER_NONE 0
#define PROFILER_MICROPROFILE 1
#define PROFILER_SUPERLUMINAL 2

//#define USE_PROFILER PROFILER_MICROPROFILE
//#define USE_PROFILER PROFILER_SUPERLUMINAL
#define USE_PROFILER PROFILER_NONE

//TODO: if (USE_PROFILER == PROFILER_MICROPROFILE)
// microprofile.cpp still compiles, inline in .h?
#pragma comment( lib, "Ws2_32" )
#pragma comment( lib, "winmm" )
//TODO: endif (USE_PROFILER == PROFILER_MICROPROFILE)

#if ( USE_PROFILER == PROFILER_MICROPROFILE )

#include <stdio.h>
#include <stdarg.h>
#include <string>
#include <thread>
#include <atomic>
//#include "unistd.h"

//note: config is in microprofile.config.h
//define MICROPROFILE_WEBSERVER_MAXFRAMES 100
#include "microprofile\microprofile.h"

//#pragma comment( lib, "Ws2_32" )
//#pragma comment( lib, "winmm" )

enum { NUM_COLS=6};
uint32_t colors[NUM_COLS] = { 0xFF6188,	0xA9DC76, 0xFFD866, 0xFC9867, 0xAB9DF2, 0x78DCE8 };
int curcol = 0;

#define PROFILE_ENTER(M) MICROPROFILE_ENTERI("defaultgroup", (M), colors[(curcol++)%NUM_COLS] )
#define PROFILE_LEAVE(M) MICROPROFILE_LEAVE()
#define PROFILE_SCOPE(M) MICROPROFILE_SCOPEI( "defaultgroup", (M), colors[(curcol++)%NUM_COLS] )
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
	printf( " done\n" );
}

//MICROPROFILE_DECLARE_LOCAL_ATOMIC_COUNTER(ThreadsStarted);
//MICROPROFILE_DEFINE_LOCAL_ATOMIC_COUNTER(ThreadSpinSleep, "/runtime/spin_sleep");
//MICROPROFILE_DECLARE_LOCAL_COUNTER(LocalCounter);
//MICROPROFILE_DEFINE_LOCAL_COUNTER(LocalCounter, "/runtime/localcounter");

#elif ( USE_PROFILER == PROFILER_SUPERLUMINAL )

#ifndef NDEBUG
#pragma comment( lib, "PerformanceAPI_MDd" )
#else
#pragma comment( lib, "PerformanceAPI_MD" )
#endif

#include <Superluminal/PerformanceAPI.h>

#define PERFORMANCEAPI_ENABLED 1

void init_profiler() {}
void deinit_profiler() {}
#define PROFILE_FUNC() PERFORMANCEAPI_INSTRUMENT_FUNCTION()
#define PROFILE_ENTER(M) PerformanceAPI::BeginEvent(M)
#define PROFILE_LEAVE(M) PerformanceAPI::EndEvent()
#define PROFILE_SCOPE(M) PERFORMANCEAPI_INSTRUMENT(M)
#define PROFILE_THREADNAME(M) PerformanceAPI::SetCurrentThreadName(M)

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

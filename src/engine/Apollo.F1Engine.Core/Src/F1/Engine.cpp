#include "F1/Engine.h"

namespace f1
{
    Engine::Engine(const EngineDescriptor &descriptor)
    {

    }

    Engine::~Engine() noexcept
    {

    }

    void Engine::Ignite()
    {

    }

    EngineHandle EngineCreate(const EngineDescriptor* pDescriptor)
    {
        return new Engine(*pDescriptor);
    }

    void EngineDestroy(EngineHandle pEngine)
    {
        delete pEngine;
    }

    void EngineIgnite(EngineHandle pEngine)
    {
        pEngine->Ignite();
    }
}
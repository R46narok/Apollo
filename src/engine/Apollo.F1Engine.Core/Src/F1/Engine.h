#ifndef F1_ENGINE_H
#define F1_ENGINE_H

#include "F1/Core.h"

namespace f1
{
    struct EngineDescriptor
    {

    };

    class Engine
    {
    public:
        explicit Engine(const EngineDescriptor& descriptor);
        ~Engine() noexcept;

        void Ignite();
    };

    extern "C"
    {
        typedef f1::Engine* EngineHandle;

        EngineHandle F1_API EngineCreate(const EngineDescriptor* pDescriptor);
        void F1_API EngineDestroy(EngineHandle pEngine);

        void F1_API EngineIgnite(EngineHandle pEngine);
    }
}

#endif //F1_ENGINE_H

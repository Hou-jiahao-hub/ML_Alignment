#pragma once

#include <map>
#include <vector>
#include <queue>
#include <mutex>
#include <stdint.h>

class AlignmentDerivBridge {
    public:
        static AlignmentDerivBridge& getInstance() {
            static AlignmentDerivBridge instance;
            return instance;
        }

        void psuhDeriv(uint64_t geoId, const std::vector<float>& deriv) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_derivMap[geoId].push(deriv);
        }

        std::vector<float> popDeriv(uint64_t geoId) {
            std::lock_guard<std::mutex> lock(m_mutex);
            if(m_derivMap.find(geoId) != m_derivMap.end() && !m_derivMap[geoId].empty()) {
                std::vector<float> val = m_derivMap[geoId].front();
                m_derivMap[geoId].pop();
                return val;
            }
            return {}
        }
    private:
        std::map<uint64_t, std::queue<std::vector<float>>> m_derivMap;
        std::mutex m_mutex;
};

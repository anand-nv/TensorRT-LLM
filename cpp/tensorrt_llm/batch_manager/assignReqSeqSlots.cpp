/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/batch_manager/assignReqSeqSlots.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"

void tensorrt_llm::batch_manager::AssignReqSeqSlots::operator()(SequenceSlotManager& seqSlotManager,
    RequestVector const& contextRequests, RequestVector const& generationRequests) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(AssignReqSeqSlots);

    seqSlotManager.freeIdleSequenceSlots();
    for (auto const& requests : {contextRequests, generationRequests})
    {
        for (auto const& llmReq : requests)
        {
            if (llmReq->isDisaggGenerationInitState())
            {
                // Skip assigning sequence slot for DISAGG_GENERATION_INIT request
                continue;
            }
            auto const isReqNew = (llmReq->isContextInitState() && llmReq->mSeqSlots.empty())
                || (llmReq->isDisaggGenerationTransmissionComplete());
            if (isReqNew && llmReq->getReturnPerfMetrics())
            {
                llmReq->setFirstScheduledTime();
            }
            for (int i = 0; i < llmReq->getNumSequences(); i++)
            {
                auto const reqSeqSlot = seqSlotManager.getSequenceSlot(isReqNew, llmReq->getSeqSlotId(i));
                TLLM_CHECK_WITH_INFO(
                    reqSeqSlot, "Unable to get batch slot for request ID %lu", lllmReq->getSeqSlotId(i));

                if ((int) llmReq->mSeqSlots.size() >= i + 1)
                {
                    llmReq->mSeqSlots[i] = reqSeqSlot.value();
                }
                else
                {
                    llmReq->mSeqSlots.push_back(reqSeqSlot.value());
                }
            }
        }
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

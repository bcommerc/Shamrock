// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ConsToPrimGasPassiveScalars.cpp
 * @author Benoît Commerçon (benoit.commercon@ens-lyon.fr)
 * @brief 
 * @date 2025-06-26
 */

#include "shammodels/ramses/modules/ConsToPrimGasPassiveScalars.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammath/riemann.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class Tvec>
    struct KernelConsToPrimGasPassiveScalars {
        using Tscal = shambase::VecComponent<Tvec>;

        inline static void kernel(
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_rho_pscal,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_rho_gas,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_pscal,
            const shambase::DistributedData<u32> &sizes,
            u32 block_size,
            u32 npscal_gas) {

            shambase::DistributedData<u32> cell_counts
                = sizes.map<u32>([&](u64 id, u32 block_count) {
                      u32 cell_count = block_count * block_size;
                      return cell_count;
                  });

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{spans_rho_pscal,spans_rho_gas},
                sham::DDMultiRef{spans_pscal},
                cell_counts,
                [npscal_gas](
                    u32 i,
                    const Tscal *__restrict rhopscal,
                    const Tscal *__restrict rhogas
                    Tscal *__restrict pscal) {
                    auto conststate_pscal = shammath::ConsStatePscal<Tvec>{rhopscal[i]};
                    auto dgas = rhogas[i];   
                    auto primstate_pscal = shammath::cons_to_prim_pscal(conststate_pscal, dgas);

                    pscal[i] = primstate_pscal.pscal;
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void NodeConsToPrimGasPassiveScalars<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();
        // Check that the size of spans_rho is compatible 
        // with the totalblock numbers for the current patch, and hydro variable on grid
        edges.spans_rho_gas.check_sizes(edges.sizes.indexes);
        edges.spans_rho_pscal.check_sizes(edges.sizes.indexes);
        edges.spans_pscal.ensure_sizes(edges.sizes.indexes);

        KernelConsToPrimPassiveScalars<Tvec>::kernel(
            edges.spans_rho_pscal.get_spans(),
            edges.spans_rho_gas.get_spans(),
            edges.spans_pscal.get_spans(),
            edges.sizes.indexes,
            block_size,
            npscal_gas);
    }

    template<class Tvec>
    std::string NodeConsToPrimGasPassiveScalars<Tvec>::_impl_get_tex() {

        auto block_count = get_ro_edge_base(0).get_tex_symbol();
        auto rho_pscal   = get_ro_edge_base(1).get_tex_symbol();
        auto rho_gas     = get_ro_edge_base(2).get_tex_symbol();
        auto pscal       = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Conservative to primitive variable (gas passive scalars)

            \begin{align}
            {pscal}_i &= \frac{ {rho_pscal}_i }{ {rho_gas}_i } \\
            i &\in [0,{block_count} * N_{\rm cell/block}) \\
            N_{\rm cell/block} & = {block_size}
            \end{align}
        )tex";

        shambase::replace_all(tex, "{rho_pscal}", rho_pscal);
        shambase::replace_all(tex, "{rho_gas}", rho_gas);
        shambase::replace_all(tex, "{pscal}", Pscal);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeConsToPrimGas<f64_3>;

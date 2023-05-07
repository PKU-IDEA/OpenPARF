#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : place_model.py
# Author            : Jing Mai <magic3007@pku.edu.cn>
# Date              : 09.29.2020
# Last Modified Date: 09.29.2020
# Last Modified By  : Jing Mai <magic3007@pku.edu.cn>

import pdb
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class PlaceModel(nn.Module):
    """
    @brief Define placement objective:
        wirelength + lambdas * density penalty
    It includes various ops related to global placement as well.
    """
    
    def __init__(self, params, placedb, data_cls, op_cls):
        super(PlaceModel, self).__init__()
        self.params = params
        self.placedb = placedb
        self.data_cls = data_cls
        self.op_cls = op_cls
    
    def obj_fn(self, pos):
        """
        @brief Compute objective.
            WL+ (lambdas^T * (phi + phi^T * D * phi)).sum()
            where lambda is a vector, phi is a vector, D is a diagonal matrix
            or we can make it simpler to element-wise expression
            WL + \sum_i lambda_i * (E_i + 0.5 * cs * E_i^2)
        @param pos locations of cells
        @return objective value
        """
        # wirelength
        wirelength = self.op_cls.wirelength_op(pos)
        
        # \sum_i lambda_i * (E_i + 0.5 * cs * E_i^2)
        phi = self.op_cls.density_op(pos)
        lambdas = self.data_cls.multiplier.lambdas
        cs = self.data_cls.multiplier.cs
        density = (phi + 0.5 * cs * phi.pow(2))
        
        # record
        self.data_cls.wirelength = wirelength.data
        self.data_cls.density = density.data
        self.data_cls.phi = phi.data
        
        # logger.debug("wirelength %g, phi %s, density %s" % (wirelength, phi.data, density.data))
        
        # record for backward
        self.wirelength = wirelength
        self.density = (lambdas * density).sum()

        obj_terms_dict = {
            "wirelength_term": self.wirelength,
            "density_term"   : self.density
        }
        
        # overall objective
        return self.wirelength + self.density, obj_terms_dict
    
    def obj_and_grad_fn(self, pos):
        """
        @brief compute objective and gradient.
            wirelength + lambdas * density penalty
        @param pos locations of cells
        @return objective value
        """
        # wirelength = self.op_cls.wirelength_op(pos)
        # if pos.grad is not None:
        #    pos.grad.zero_()
        # wirelength.backward()
        # wirelength_grad = pos.grad.data.clone()
        ##print("|wl| before precond inst  ", wirelength_grad[self.data_cls.area_type_inst_groups[3][:100]].norm(p=1))
        ##print("|wl| before precond filler", wirelength_grad[self.data_cls.area_type_inst_groups[3][100:]].norm(p=1))
        ##print("|wl| before precond", wirelength_grad[self.data_cls.area_type_inst_groups[3]].norm(p=1))
        ##self.op_cls.precond_op(wirelength_grad)
        ##print("|wl| after precond inst  ", wirelength_grad[self.data_cls.area_type_inst_groups[3][:100]].norm(p=1))
        ##print("|wl| after precond filler", wirelength_grad[self.data_cls.area_type_inst_groups[3][100:]].norm(p=1))
        ##print("|wl| after precond", wirelength_grad[self.data_cls.area_type_inst_groups[3]].norm(p=1))
        
        # phi = self.op_cls.density_op(pos)
        ## compute density term
        # if pos.grad is not None:
        # density = (phi + 0.5 * self.data_cls.multiplier.cs * phi.pow(2))
        # density = (self.data_cls.multiplier.lambdas * density).sum()
        #    pos.grad.zero_()
        # density.backward()
        # density_grad = pos.grad.data.clone()
        ##print("|D| before precond inst  ", density_grad[self.data_cls.area_type_inst_groups[3][:100]].norm(p=1))
        ##print("|D| before precond filler", density_grad[self.data_cls.area_type_inst_groups[3][100:]].norm(p=1))
        ##print("|D| before precond", density_grad[self.data_cls.area_type_inst_groups[3]].norm(p=1))
        ##self.op_cls.precond_op(density_grad)
        ##print("|D| after precond inst  ", density_grad[self.data_cls.area_type_inst_groups[3][:100]].norm(p=1))
        ##print("|D| after precond filler", density_grad[self.data_cls.area_type_inst_groups[3][100:]].norm(p=1))
        ##print("|D| after precond", density_grad[self.data_cls.area_type_inst_groups[3]].norm(p=1))
        
        # ratios = [None] * self.data_cls.num_area_types
        # for at in range(self.data_cls.num_area_types - 1):
        #    inst_ids = self.data_cls.area_type_inst_groups[at]
        #    if len(inst_ids):
        #        wirelength_grad_norm = wirelength_grad[inst_ids].norm(p=1)
        #        density_grad_norm = density_grad[inst_ids].norm(p=1)
        #        ratios[at] = density_grad_norm / wirelength_grad_norm
        #        #print("b?? at %d, |wl grad| = %g, |D grad| = %g, |D/wl| = %g" % (at, wirelength_grad_norm, density_grad_norm, ratios[at]))
        
        ## compute normalization factor alphas for preconditioning
        # alphas = pos.new_ones(self.data_cls.num_area_types)
        # for area_type in range(self.data_cls.num_area_types):
        #    inst_ids = self.data_cls.area_type_inst_groups[area_type]
        #    if len(inst_ids):
        #        wirelength_grad_norm = wirelength_grad[inst_ids].norm(p=1)
        #        density_grad_norm = density_grad[inst_ids].norm(p=1)
        #        if wirelength_grad_norm > 0:
        #            alphas[area_type] = density_grad_norm / wirelength_grad_norm
        # alphas = self.op_cls.stable_zero_div_op(alphas.clamp_(min=1.0), self.data_cls.multiplier.lambdas)
        ##alphas = pos.new_ones(self.data_cls.num_area_types)
        ##for at in range(self.data_cls.num_area_types - 1):
        ##    inst_ids = self.data_cls.area_type_inst_groups[at]
        ##    if at == 3 and ratios[at]:
        ##        #wl_precond_mean = self.data_cls.wl_precond[inst_ids[:100]].mean()
        ##        inst_gw = wirelength_grad[inst_ids[:100]].norm(p=1)
        ##        inst_gd = density_grad[inst_ids[:100]].norm(p=1)
        ##        filler_gd = (density_grad[inst_ids[100:]] / self.data_cls.multiplier.lambdas[at] / self.data_cls.inst_areas[inst_ids[100:], at].view([-1, 1])).norm(p=1)
        ##        #alphas[at] = (inst_gw - inst_gd) / filler_gd - self.data_cls.multiplier.lambdas[at] * self.data_cls.inst_areas[inst_ids[:100], at].mean()
        ##        #alphas[at] = (wl_precond_mean / alphas[at]).clamp_(min=1.0).item()
        ##        alphas[at] = (filler_gd / inst_gw)
        ##        #alphas[at] = ratios[at].item()
        # print("alphas = ", alphas)
        
        # wirelength = self.op_cls.wirelength_op(pos)
        # if pos.grad is not None:
        #    pos.grad.zero_()
        # wirelength.backward()
        # wirelength_grad = pos.grad.clone()
        # print("|wl| before precond inst  ", wirelength_grad[self.data_cls.area_type_inst_groups[3][:100]].norm(p=1))
        # print("|wl| before precond filler", wirelength_grad[self.data_cls.area_type_inst_groups[3][100:]].norm(p=1))
        ##print("|wl| before precond", wirelength_grad[self.data_cls.area_type_inst_groups[3]].norm(p=1))
        # self.op_cls.precond_op(wirelength_grad, alphas)
        # print("|wl| after precond inst  ", wirelength_grad[self.data_cls.area_type_inst_groups[3][:100]].norm(p=1))
        # print("|wl| after precond filler", wirelength_grad[self.data_cls.area_type_inst_groups[3][100:]].norm(p=1))
        ##print("|wl| after precond", wirelength_grad[self.data_cls.area_type_inst_groups[3]].norm(p=1))
        
        # phi = self.op_cls.density_op(pos)
        ## compute density term
        # density = (phi + 0.5 * self.data_cls.multiplier.cs * phi.pow(2))
        # density = (self.data_cls.multiplier.lambdas * density).sum()
        # if pos.grad is not None:
        #    pos.grad.zero_()
        # density.backward()
        # density_grad = pos.grad.clone()
        # print("|D| before precond inst  ", density_grad[self.data_cls.area_type_inst_groups[3][:100]].norm(p=1))
        # print("|D| before precond filler", density_grad[self.data_cls.area_type_inst_groups[3][100:]].norm(p=1))
        ##print("|D| before precond", density_grad[self.data_cls.area_type_inst_groups[3]].norm(p=1))
        # self.op_cls.precond_op(density_grad, alphas)
        # print("|D| after precond inst  ", density_grad[self.data_cls.area_type_inst_groups[3][:100]].norm(p=1))
        # print("|D| after precond filler", density_grad[self.data_cls.area_type_inst_groups[3][100:]].norm(p=1))
        ##print("|D| after precond", density_grad[self.data_cls.area_type_inst_groups[3]].norm(p=1))
        
        # ratios = [None] * self.data_cls.num_area_types
        # for at in range(self.data_cls.num_area_types - 1):
        #    inst_ids = self.data_cls.area_type_inst_groups[at]
        #    if len(inst_ids):
        #        wirelength_grad_norm = wirelength_grad[inst_ids].norm(p=1)
        #        density_grad_norm = density_grad[inst_ids].norm(p=1)
        #        ratios[at] = density_grad_norm / wirelength_grad_norm
        #        print("a!! at %d, |wl grad| = %g, |D grad| = %g, |D/wl| = %g" % (at, wirelength_grad_norm, density_grad_norm, ratios[at]))
        
        obj, _ = self.obj_fn(pos)
        
        if pos.grad is not None:
            pos.grad.zero_()
        
        self.wirelength.backward()
        wirelength_grad = pos.grad.data.clone()
        
        if pos.grad is not None:
            pos.grad.zero_()
        
        self.density.backward()
        density_grad = pos.grad.data.clone()
        
        # overall gradient
        pos.grad.data.copy_(wirelength_grad + density_grad)
        # in case some instances are locked
        pos.grad.data.masked_fill_(self.data_cls.inst_lock_mask.view([-1, 1]), 0)
        
        if self.params.gp_dynamic_precondition: 
            # compute normalization factor alphas for preconditioning
            gd_gw_norm_ratios = pos.new_ones(self.data_cls.num_area_types)
            for area_type in range(self.data_cls.num_area_types - 1):
                inst_ids = self.data_cls.area_type_inst_groups[area_type]
                if len(inst_ids):
                    wirelength_grad_norm = wirelength_grad[inst_ids].norm(p=1)
                    density_grad_norm = density_grad[inst_ids].norm(p=1)
                    gd_gw_norm_ratios[area_type] = density_grad_norm / wirelength_grad_norm
            self.data_cls.multiplier.gd_gw_norm_ratio = self.op_cls.stable_zero_div_op(gd_gw_norm_ratios.clamp_(min=1.0),
                                                                                    self.data_cls.multiplier.lambdas)
            precond_alphas = self.data_cls.multiplier.gd_gw_norm_ratio
        else:
            precond_alphas = pos.new_ones(self.data_cls.num_area_types)
        
        self.op_cls.precond_op(pos.grad, precond_alphas)
        
        grad_dicts = {
            'wirelength_grad_norm': wirelength_grad.norm(p=1),
            'density_grad_norm'   : density_grad.norm(p=1)
        }
        return obj, pos.grad, grad_dicts
    
    def forward(self):
        """
        @brief Compute objective with current locations of cells.
        """
        return self.obj_fn(self.data_cls.pos[0])[0]


class FenceRegionPlaceModel(PlaceModel):
    def __init__(self, params, placedb, data_cls, op_cls):
        super(FenceRegionPlaceModel, self).__init__(params, placedb, data_cls, op_cls)
        self.fence_region_cost_term = None
    
    def obj_fn(self, pos):
        wirelength_and_density_obj, obj_terms_dict = super(FenceRegionPlaceModel, self).obj_fn(pos)
        
        # fence region cost. Note that fence region cost only makes senses to movable instances.
        movable_pos = pos[self.data_cls.movable_range[0]: self.data_cls.movable_range[1]]
        fence_region_cost = self.op_cls.fence_region_op(movable_pos)
        eta = self.data_cls.fence_region_cost_parameters.eta
        
        # record
        self.data_cls.fence_region_cost = fence_region_cost.data
        
        # record for backward
        self.fence_region_cost_term = (eta * fence_region_cost).sum()
       
        obj_terms_dict['fence_region_term'] = self.fence_region_cost_term
        
        return wirelength_and_density_obj + self.fence_region_cost_term, obj_terms_dict
    
    def obj_and_grad_fn(self, pos):
        obj, _ = self.obj_fn(pos)
        
        if pos.grad is not None:
            pos.grad.zero_()
        self.wirelength.backward()
        wirelength_grad = pos.grad.data.clone()
        
        if pos.grad is not None:
            pos.grad.zero_()
        self.density.backward()
        density_grad = pos.grad.data.clone()
        
        # Note that fence region cost only makes senses to movable instances.
        if pos.grad is not None:
            pos.grad.zero_()
        self.fence_region_cost_term.backward()
        fence_region_cost_term_grad = pos.grad.data.clone()[self.data_cls.movable_range[0]
                                                            :self.data_cls.movable_range[1]]
        
        # overall gradient
        pos.grad.data.copy_(wirelength_grad + density_grad)
       
        if self.params.gp_dynamic_precondition: 
            # compute normalization factor alphas for preconditioning
            gd_gw_norm_ratios = pos.new_ones(self.data_cls.num_area_types)
            for area_type in range(self.data_cls.num_area_types - 1):
                inst_ids = self.data_cls.area_type_inst_groups[area_type]
                if len(inst_ids):
                    wirelength_grad_norm = wirelength_grad[inst_ids].norm(p=1)
                    density_grad_norm = density_grad[inst_ids].norm(p=1)
                    gd_gw_norm_ratios[area_type] = density_grad_norm / wirelength_grad_norm
            self.data_cls.multiplier.gd_gw_norm_ratio = self.op_cls.stable_zero_div_op(gd_gw_norm_ratios.clamp_(min=1.0),
                                                                                    self.data_cls.multiplier.lambdas)
            precond_alphas = self.data_cls.multiplier.gd_gw_norm_ratio
        else:
            precond_alphas = pos.new_ones(self.data_cls.num_area_types)
            
        self.op_cls.precond_op(pos.grad, precond_alphas)
        
        temp_grad = pos.grad.data.clone()
        temp_grad[self.data_cls.movable_range[0]:self.data_cls.movable_range[1]] += fence_region_cost_term_grad
        pos.grad.data.copy_(temp_grad)

        # in case some instances are locked
        pos.grad.data.masked_fill_(self.data_cls.inst_lock_mask.view([-1, 1]), 0)

        # area_type_mover_density_grad_norm = pos.new_ones(self.data_cls.num_area_types)
        # area_type_filler_density_grad_norm = pos.new_ones(self.data_cls.num_area_types)
        # area_type_density_grad_norm = pos.new_ones(self.data_cls.num_area_types)
        # for area_type in range(self.data_cls.num_area_types - 1):
        #     inst_ids = self.data_cls.area_type_inst_groups[area_type]
        #     if len(inst_ids):
        #         area_type_density_grad_norm[area_type] = density_grad[inst_ids].norm(p=1)
        #     movable_inst_ids = inst_ids[
        #         (self.data_cls.movable_range[0] <= inst_ids) * (inst_ids < self.data_cls.movable_range[1])]
        #     filler_inst_ids = inst_ids[
        #         (self.data_cls.filler_range[0] <= inst_ids) * (inst_ids < self.data_cls.filler_range[1])]
        #     if len(movable_inst_ids):
        #         area_type_mover_density_grad_norm[area_type] = density_grad[movable_inst_ids].norm(p=1) / len(movable_inst_ids)
        #     if len(filler_inst_ids):
        #         area_type_filler_density_grad_norm[area_type] = density_grad[filler_inst_ids].norm(p=1) / len(filler_inst_ids)
        # area_type_density_grad_norm /= wirelength_grad.norm(p=1)
        # logger.info("area_type_density_grad_norm: ")
        # logger.info(area_type_density_grad_norm.numpy())
        # logger.info("area_type_mover_density_grad_norm(avg per instance): ")
        # logger.info(area_type_mover_density_grad_norm.numpy())
        # logger.info("area_type_filler_density_grad_norm(avg per instance): ")
        # logger.info(area_type_filler_density_grad_norm.numpy())
        # logger.info("gd_gw_norm_ratio:")
        # logger.info(self.data_cls.multiplier.gd_gw_norm_ratio.numpy())
        
        grad_dicts = {
            'wirelength_grad_norm'  : wirelength_grad.norm(p=1),
            'density_grad_norm'     : density_grad.norm(p=1),
            'fence_region_grad_norm': fence_region_cost_term_grad.norm(p=1)
        }
        
        return obj, pos.grad, grad_dicts
    
    def forward(self):
        return self.obj_fn(self.data_cls.pos[0])[0]

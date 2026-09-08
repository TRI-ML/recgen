import json
import os
import yaml

import pandas as pd

import numpy as np

from ..core.metrics import (compute_RT_distances, compute_add, compute_adds, chamfer_distance_gt_mesh,
    compute_add_gen, compute_adds_gen, chamfer_distance_gt_mesh_color)


class Evaluator:
    def __init__(self, config="../configs/eval/configs.yaml"):
        # Initialize other necessary attributes
        self.results = {}
        self.metrics = {}
        self.all_frames_metrics = {'ADD': [], 'ADD-S': [], 'AR': [],
                                   'VSD': [], 'MSSD': [], 'MSPD': [],
                                   #'VSD_rec': [], 'MSSD_rec': [], 'MSPD_rec': [],
                                   'CHAMFER': [], 'CHAMFER_NO_ICP': [], 'CHAMFER_ICP_WITH_SCALE': [],
                                   'CHAMFER_color': [], 'CHAMFER_color_NO_ICP': [],
                                   'chamfer_normalized': [], 'chamfer_normalized_icp_with_scale': [], 'ADDSS': [],
                                   'ADDSS_10': [], 'ADDSS_05': [], 'ADDSS_02': [],
                                   'LPIPS': [], 'LPIPS_NO_ICP': [],
                                   'SSIM': [], 'SSIM_NO_ICP': [],
                                   'PSNR': [], 'PSNR_NO_ICP': [],
                                   'COLOR_DIST': [], 'COLOR_DIST_NO_ICP': [],
                                   'R_error': [], 'T_error': [],
                                   'Frame_ID': [], 'Class': [],
                                   'FAILED': []}
        # Track failed samples per object and globally
        self.failures = {}  # {object_id: [file_names]}
        self.total_failures = 0
        self.config = config

        # Load configuration from the provided path if needed
        with open(self.config, 'r') as f:
            self.config_data = yaml.safe_load(f)

        #self.pose_recall_th = self.config_data["pose_recall_th"]
        self.mssd_rec = np.array(self.config_data["mssd_rec"])
        self.mspd_rec = np.array(self.config_data["mspd_rec"])
        self.vsd_delta = self.config_data["vsd_delta"]
        self.vsd_taus = self.config_data["vsd_taus"]
        self.vsd_rec = np.array(self.config_data["vsd_rec"])
        self.filter_outliers_type = self.config_data["filter_outliers_type"]
        self.filter_outliers_params = self.config_data["filter_outliers_params"]

        # Mask erosion config (with backwards-compatible defaults)
        self.mask_erosion_enabled = self.config_data.get("mask_erosion_enabled", True)
        self.mask_erosion_params = self.config_data.get("mask_erosion_params", {
            "kernel_size": 5,
            "iterations": 1
        })

        # Pointmap normalization config (with backwards-compatible default)
        self.normalization_method = self.config_data.get("normalization_method", "minmax")
        self.quantile_drop_threshold = self.config_data.get("quantile_drop_threshold", 0.025)

        # Color chamfer config (disabled by default — slow)
        self.compute_color_chamfer = self.config_data.get("compute_color_chamfer", False)



    def add_object_to_metrics(self, object_id):
        if object_id not in self.metrics:
            self.metrics[object_id] = {'ADD': [], 'ADD-S': [], 'AR': [],
                                       'VSD': [], 'MSSD': [], 'MSPD': [],
                                       'VSD_rec': [], 'MSSD_rec': [], 'MSPD_rec': [],
                                       'CHAMFER': [], 'CHAMFER_NO_ICP': [], 'CHAMFER_ICP_WITH_SCALE': [],
                                       'CHAMFER_color': [], 'CHAMFER_color_NO_ICP': [],
                                       'chamfer_normalized': [], 'chamfer_normalized_icp_with_scale': [], 'ADDSS': [],
                                       'ADDSS_10': [], 'ADDSS_05': [], 'ADDSS_02': [],
                                       'LPIPS': [], 'LPIPS_NO_ICP': [],
                                       'SSIM': [], 'SSIM_NO_ICP': [],
                                       'PSNR': [], 'PSNR_NO_ICP': [],
                                       'COLOR_DIST': [], 'COLOR_DIST_NO_ICP': [],
                                       'R_error': [], 'T_error': [], 'cls_id': [], 'instance_id': []}

    def add_metrics(self, object_id, metrics, instance_id, failed=False):
        if object_id not in self.metrics:
            self.add_object_to_metrics(object_id)

        frame_id = str(int(instance_id.split('.')[0]))
        self.metrics[object_id]['ADD'].append(metrics['add_thres'])
        self.metrics[object_id]['ADD-S'].append(metrics['adds_thres'])
        self.metrics[object_id]['AR'].append(metrics['mean_ar'])
        self.metrics[object_id]['VSD'].append(metrics['mean_vsd'])
        self.metrics[object_id]['MSSD'].append(metrics['mean_mssd'])
        self.metrics[object_id]['MSPD'].append(metrics['mean_mspd'])
        self.metrics[object_id]['VSD_rec'].append(metrics['vsd_rec'])
        self.metrics[object_id]['MSSD_rec'].append(metrics['mssd_rec'])
        self.metrics[object_id]['MSPD_rec'].append(metrics['mspd_rec'])
        self.metrics[object_id]['R_error'].append(metrics['err_R'])
        self.metrics[object_id]['T_error'].append(metrics['err_T'])
        self.metrics[object_id]['cls_id'].append(object_id)
        self.metrics[object_id]['instance_id'].append(frame_id)
        if 'chamfer_dist' in metrics:
            self.metrics[object_id]['CHAMFER'].append(metrics['chamfer_dist'])
        if 'chamfer_dist_no_icp' in metrics:
            self.metrics[object_id]['CHAMFER_NO_ICP'].append(metrics['chamfer_dist_no_icp'])
        if 'chamfer_dist_icp_with_scale' in metrics:
            self.metrics[object_id]['CHAMFER_ICP_WITH_SCALE'].append(metrics['chamfer_dist_icp_with_scale'])
        if 'chamfer_dist_color' in metrics and not np.isnan(metrics['chamfer_dist_color']):
            self.metrics[object_id]['CHAMFER_color'].append(metrics['chamfer_dist_color'])
        if 'chamfer_dist_color_no_icp' in metrics and not np.isnan(metrics['chamfer_dist_color_no_icp']):
            self.metrics[object_id]['CHAMFER_color_NO_ICP'].append(metrics['chamfer_dist_color_no_icp'])
        self.metrics[object_id]['chamfer_normalized'].append(metrics.get('chamfer_normalized', np.nan))
        self.metrics[object_id]['chamfer_normalized_icp_with_scale'].append(metrics.get('chamfer_normalized_icp_with_scale', np.nan))
        self.metrics[object_id]['ADDSS'].append(metrics.get('addss', np.nan))
        self.metrics[object_id]['ADDSS_10'].append(metrics.get('addss_10', np.nan))
        self.metrics[object_id]['ADDSS_05'].append(metrics.get('addss_05', np.nan))
        self.metrics[object_id]['ADDSS_02'].append(metrics.get('addss_02', np.nan))
        # Perception metrics (optional — only present when computed)
        _PM_KEYS = [('LPIPS', 'lpips'), ('LPIPS_NO_ICP', 'lpips_no_icp'),
                     ('SSIM', 'ssim'), ('SSIM_NO_ICP', 'ssim_no_icp'),
                     ('PSNR', 'psnr'), ('PSNR_NO_ICP', 'psnr_no_icp'),
                     ('COLOR_DIST', 'color_dist'), ('COLOR_DIST_NO_ICP', 'color_dist_no_icp')]
        for col_key, metric_key in _PM_KEYS:
            self.metrics[object_id][col_key].append(metrics.get(metric_key, np.nan))

        self.all_frames_metrics['ADD'].append(metrics['add_thres'])
        self.all_frames_metrics['ADD-S'].append(metrics['adds_thres'])
        self.all_frames_metrics['AR'].append(metrics['mean_ar'])
        self.all_frames_metrics['VSD'].append(metrics['mean_vsd'])
        self.all_frames_metrics['MSSD'].append(metrics['mean_mssd'])
        self.all_frames_metrics['MSPD'].append(metrics['mean_mspd'])
        #self.all_frames_metrics['VSD_rec'].append(metrics['vsd_rec'])
        #self.all_frames_metrics['MSSD_rec'].append(metrics['mssd_rec'].tolist())
        #self.all_frames_metrics['MSPD_rec'].append(metrics['mspd_rec'].tolist())
        self.all_frames_metrics['R_error'].append(metrics['err_R'])
        self.all_frames_metrics['T_error'].append(metrics['err_T'])
        self.all_frames_metrics['Class'].append(object_id)
        self.all_frames_metrics['Frame_ID'].append(frame_id)
        self.all_frames_metrics['FAILED'].append(1 if failed else 0)
        self.all_frames_metrics['CHAMFER'].append(metrics.get('chamfer_dist', np.nan))
        self.all_frames_metrics['CHAMFER_NO_ICP'].append(metrics.get('chamfer_dist_no_icp', np.nan))
        self.all_frames_metrics['CHAMFER_ICP_WITH_SCALE'].append(metrics.get('chamfer_dist_icp_with_scale', np.nan))
        chamfer_color = metrics.get('chamfer_dist_color', np.nan)
        self.all_frames_metrics['CHAMFER_color'].append(chamfer_color if not np.isnan(chamfer_color) else np.nan)
        chamfer_color_no_icp = metrics.get('chamfer_dist_color_no_icp', np.nan)
        self.all_frames_metrics['CHAMFER_color_NO_ICP'].append(chamfer_color_no_icp if not np.isnan(chamfer_color_no_icp) else np.nan)
        self.all_frames_metrics['chamfer_normalized'].append(metrics.get('chamfer_normalized', np.nan))
        self.all_frames_metrics['chamfer_normalized_icp_with_scale'].append(metrics.get('chamfer_normalized_icp_with_scale', np.nan))
        self.all_frames_metrics['ADDSS'].append(metrics.get('addss', np.nan))
        self.all_frames_metrics['ADDSS_10'].append(metrics.get('addss_10', np.nan))
        self.all_frames_metrics['ADDSS_05'].append(metrics.get('addss_05', np.nan))
        self.all_frames_metrics['ADDSS_02'].append(metrics.get('addss_02', np.nan))
        for col_key, metric_key in _PM_KEYS:
            self.all_frames_metrics[col_key].append(metrics.get(metric_key, np.nan))

        if failed:
            if object_id not in self.failures:
                self.failures[object_id] = []
            self.failures[object_id].append(instance_id)
            self.total_failures += 1


    def get_object_metrics(self, obj_id):
        df_obj = pd.DataFrame({
            'Frame_ID': self.metrics[obj_id]['instance_id'],
            'Class': self.metrics[obj_id]['cls_id'],
            'ADD-S': self.metrics[obj_id]['ADD-S'],
            'ADD': self.metrics[obj_id]['ADD'],
            'AR': self.metrics[obj_id]['AR'],
            'MSSD': self.metrics[obj_id]['MSSD'],
            'MSPD': self.metrics[obj_id]['MSPD'],
            'VSD': self.metrics[obj_id]['VSD'],
            'R_error': self.metrics[obj_id]['R_error'],
            'T_error': self.metrics[obj_id]['T_error'],
        })
        if 'CHAMFER' in self.metrics[obj_id] and len(self.metrics[obj_id]['CHAMFER']) == len(df_obj):
            df_obj['CHAMFER'] = self.metrics[obj_id]['CHAMFER']
        if 'CHAMFER_NO_ICP' in self.metrics[obj_id] and len(self.metrics[obj_id]['CHAMFER_NO_ICP']) == len(df_obj):
            df_obj['CHAMFER_NO_ICP'] = self.metrics[obj_id]['CHAMFER_NO_ICP']
        if 'CHAMFER_ICP_WITH_SCALE' in self.metrics[obj_id] and len(self.metrics[obj_id]['CHAMFER_ICP_WITH_SCALE']) == len(df_obj):
            df_obj['CHAMFER_ICP_WITH_SCALE'] = self.metrics[obj_id]['CHAMFER_ICP_WITH_SCALE']
        if 'CHAMFER_color' in self.metrics[obj_id] and len(self.metrics[obj_id]['CHAMFER_color']) == len(df_obj):
            df_obj['CHAMFER_color'] = self.metrics[obj_id]['CHAMFER_color']
        if 'CHAMFER_color_NO_ICP' in self.metrics[obj_id] and len(self.metrics[obj_id]['CHAMFER_color_NO_ICP']) == len(df_obj):
            df_obj['CHAMFER_color_NO_ICP'] = self.metrics[obj_id]['CHAMFER_color_NO_ICP']
        if 'chamfer_normalized' in self.metrics[obj_id] and len(self.metrics[obj_id]['chamfer_normalized']) == len(df_obj):
            df_obj['chamfer_normalized'] = self.metrics[obj_id]['chamfer_normalized']
        if 'chamfer_normalized_icp_with_scale' in self.metrics[obj_id] and len(self.metrics[obj_id]['chamfer_normalized_icp_with_scale']) == len(df_obj):
            df_obj['chamfer_normalized_icp_with_scale'] = self.metrics[obj_id]['chamfer_normalized_icp_with_scale']
        if 'ADDSS' in self.metrics[obj_id] and len(self.metrics[obj_id]['ADDSS']) == len(df_obj):
            df_obj['ADDSS'] = self.metrics[obj_id]['ADDSS']
        if 'ADDSS_10' in self.metrics[obj_id] and len(self.metrics[obj_id]['ADDSS_10']) == len(df_obj):
            df_obj['ADDSS_10'] = self.metrics[obj_id]['ADDSS_10']
        if 'ADDSS_05' in self.metrics[obj_id] and len(self.metrics[obj_id]['ADDSS_05']) == len(df_obj):
            df_obj['ADDSS_05'] = self.metrics[obj_id]['ADDSS_05']
        if 'ADDSS_02' in self.metrics[obj_id] and len(self.metrics[obj_id]['ADDSS_02']) == len(df_obj):
            df_obj['ADDSS_02'] = self.metrics[obj_id]['ADDSS_02']
        for pm_key in ['LPIPS', 'LPIPS_NO_ICP', 'SSIM', 'SSIM_NO_ICP', 'PSNR', 'PSNR_NO_ICP', 'COLOR_DIST', 'COLOR_DIST_NO_ICP']:
            if pm_key in self.metrics[obj_id] and len(self.metrics[obj_id][pm_key]) == len(df_obj):
                df_obj[pm_key] = self.metrics[obj_id][pm_key]

        means_all = {
            'ADD-S': np.mean(self.metrics[obj_id]['ADD-S']) * 100,
            'ADD': np.mean(self.metrics[obj_id]['ADD']) * 100,
            'AR': np.mean(self.metrics[obj_id]['AR']) * 100,
            'MSSD': np.mean(self.metrics[obj_id]['MSSD']) * 100,
            'MSPD': np.mean(self.metrics[obj_id]['MSPD']) * 100,
            'VSD': np.mean(self.metrics[obj_id]['VSD']) * 100,
            'R_error': np.mean(self.metrics[obj_id]['R_error']),
            'T_error': np.mean(self.metrics[obj_id]['T_error']),
        }
        if 'CHAMFER' in self.metrics[obj_id]:
            means_all['CHAMFER'] = np.mean(self.metrics[obj_id]['CHAMFER'])
        if 'CHAMFER_NO_ICP' in self.metrics[obj_id] and self.metrics[obj_id]['CHAMFER_NO_ICP']:
            means_all['CHAMFER_NO_ICP'] = np.mean(self.metrics[obj_id]['CHAMFER_NO_ICP'])
        if 'CHAMFER_ICP_WITH_SCALE' in self.metrics[obj_id] and self.metrics[obj_id]['CHAMFER_ICP_WITH_SCALE']:
            means_all['CHAMFER_ICP_WITH_SCALE'] = np.mean(self.metrics[obj_id]['CHAMFER_ICP_WITH_SCALE'])
        if 'CHAMFER_color' in self.metrics[obj_id] and self.metrics[obj_id]['CHAMFER_color']:
            means_all['CHAMFER_color'] = np.mean(self.metrics[obj_id]['CHAMFER_color'])
        if 'CHAMFER_color_NO_ICP' in self.metrics[obj_id] and self.metrics[obj_id]['CHAMFER_color_NO_ICP']:
            means_all['CHAMFER_color_NO_ICP'] = np.mean(self.metrics[obj_id]['CHAMFER_color_NO_ICP'])
        if 'chamfer_normalized' in self.metrics[obj_id] and self.metrics[obj_id]['chamfer_normalized']:
            means_all['chamfer_normalized'] = np.nanmean(self.metrics[obj_id]['chamfer_normalized'])
        if 'chamfer_normalized_icp_with_scale' in self.metrics[obj_id] and self.metrics[obj_id]['chamfer_normalized_icp_with_scale']:
            means_all['chamfer_normalized_icp_with_scale'] = np.nanmean(self.metrics[obj_id]['chamfer_normalized_icp_with_scale'])
        if 'ADDSS' in self.metrics[obj_id] and self.metrics[obj_id]['ADDSS']:
            means_all['ADDSS'] = np.nanmean(self.metrics[obj_id]['ADDSS'])
        if 'ADDSS_10' in self.metrics[obj_id] and self.metrics[obj_id]['ADDSS_10']:
            means_all['ADDSS_10'] = np.nanmean(self.metrics[obj_id]['ADDSS_10'])
        if 'ADDSS_05' in self.metrics[obj_id] and self.metrics[obj_id]['ADDSS_05']:
            means_all['ADDSS_05'] = np.nanmean(self.metrics[obj_id]['ADDSS_05'])
        if 'ADDSS_02' in self.metrics[obj_id] and self.metrics[obj_id]['ADDSS_02']:
            means_all['ADDSS_02'] = np.nanmean(self.metrics[obj_id]['ADDSS_02'])
        for pm_key in ['LPIPS', 'LPIPS_NO_ICP', 'SSIM', 'SSIM_NO_ICP', 'PSNR', 'PSNR_NO_ICP', 'COLOR_DIST', 'COLOR_DIST_NO_ICP']:
            if pm_key in self.metrics[obj_id] and self.metrics[obj_id][pm_key]:
                means_all[pm_key] = np.nanmean(self.metrics[obj_id][pm_key])

        mean_row_df = pd.DataFrame({
            'Frame_ID': ['MEAN'],
            'Class': [obj_id],
            'ADD-S': [f"{means_all['ADD-S']:.1f}"],
            'ADD': [f"{means_all['ADD']:.1f}"],
            'AR': [f"{means_all['AR']:.1f}"],
            'MSSD': [f"{means_all['MSSD']:.1f}"],
            'MSPD': [f"{means_all['MSPD']:.1f}"],
            'VSD': [f"{means_all['VSD']:.1f}"],
            'R_error': [f"{means_all['R_error']:.1f}"],
            'T_error': [f"{means_all['T_error']:.1f}"]
        })
        if 'CHAMFER' in self.metrics[obj_id]:
            mean_row_df['CHAMFER'] = [f"{means_all['CHAMFER']:.1f}"]
        if 'CHAMFER_NO_ICP' in means_all:
            mean_row_df['CHAMFER_NO_ICP'] = [f"{means_all['CHAMFER_NO_ICP']:.1f}"]
        if 'CHAMFER_ICP_WITH_SCALE' in means_all:
            mean_row_df['CHAMFER_ICP_WITH_SCALE'] = [f"{means_all['CHAMFER_ICP_WITH_SCALE']:.1f}"]
        if 'CHAMFER_color' in means_all:
            mean_row_df['CHAMFER_color'] = [f"{means_all['CHAMFER_color']:.1f}"]
        if 'CHAMFER_color_NO_ICP' in means_all:
            mean_row_df['CHAMFER_color_NO_ICP'] = [f"{means_all['CHAMFER_color_NO_ICP']:.1f}"]
        if 'chamfer_normalized' in means_all:
            mean_row_df['chamfer_normalized'] = [f"{means_all['chamfer_normalized']:.4f}"]
        if 'chamfer_normalized_icp_with_scale' in means_all:
            mean_row_df['chamfer_normalized_icp_with_scale'] = [f"{means_all['chamfer_normalized_icp_with_scale']:.4f}"]
        if 'ADDSS' in means_all:
            mean_row_df['ADDSS'] = [f"{means_all['ADDSS']:.4f}"]
        if 'ADDSS_10' in means_all:
            mean_row_df['ADDSS_10'] = [f"{means_all['ADDSS_10']:.4f}"]
        if 'ADDSS_05' in means_all:
            mean_row_df['ADDSS_05'] = [f"{means_all['ADDSS_05']:.4f}"]
        if 'ADDSS_02' in means_all:
            mean_row_df['ADDSS_02'] = [f"{means_all['ADDSS_02']:.4f}"]
        for pm_key in ['LPIPS', 'LPIPS_NO_ICP', 'SSIM', 'SSIM_NO_ICP', 'PSNR', 'PSNR_NO_ICP', 'COLOR_DIST', 'COLOR_DIST_NO_ICP']:
            if pm_key in means_all:
                mean_row_df[pm_key] = [f"{means_all[pm_key]:.4f}"]

        df_obj = pd.concat([df_obj, mean_row_df], ignore_index=True)

        num_failures = len(self.failures.get(obj_id, []))
        num_total = len(self.metrics[obj_id]['ADD-S']) + num_failures

        row_data = {
            'Class_ID': obj_id,
            'N_total': num_total,
            'N_failed': num_failures,
            'ADD-S': f"{means_all['ADD-S']:.1f}",
            'ADD': f"{means_all['ADD']:.1f}",
            'AR': f"{means_all['AR']:.1f}",
            'MSSD': f"{means_all['MSSD']:.1f}",
            'MSPD': f"{means_all['MSPD']:.1f}",
            'VSD': f"{means_all['VSD']:.1f}",
        }
        if 'CHAMFER' in means_all:
            row_data['CHAMFER'] = f"{means_all['CHAMFER']:.1f}"
        if 'CHAMFER_NO_ICP' in means_all:
            row_data['CHAMFER_NO_ICP'] = f"{means_all['CHAMFER_NO_ICP']:.1f}"
        if 'CHAMFER_ICP_WITH_SCALE' in means_all:
            row_data['CHAMFER_ICP_WITH_SCALE'] = f"{means_all['CHAMFER_ICP_WITH_SCALE']:.1f}"
        if 'CHAMFER_color' in means_all:
            row_data['CHAMFER_color'] = f"{means_all['CHAMFER_color']:.1f}"
        if 'CHAMFER_color_NO_ICP' in means_all:
            row_data['CHAMFER_color_NO_ICP'] = f"{means_all['CHAMFER_color_NO_ICP']:.1f}"
        if 'chamfer_normalized' in means_all:
            row_data['chamfer_normalized'] = f"{means_all['chamfer_normalized']:.4f}"
        if 'chamfer_normalized_icp_with_scale' in means_all:
            row_data['chamfer_normalized_icp_with_scale'] = f"{means_all['chamfer_normalized_icp_with_scale']:.4f}"
        if 'ADDSS' in means_all:
            row_data['ADDSS'] = f"{means_all['ADDSS']:.4f}"
        if 'ADDSS_10' in means_all:
            row_data['ADDSS_10'] = f"{means_all['ADDSS_10']:.4f}"
        if 'ADDSS_05' in means_all:
            row_data['ADDSS_05'] = f"{means_all['ADDSS_05']:.4f}"
        if 'ADDSS_02' in means_all:
            row_data['ADDSS_02'] = f"{means_all['ADDSS_02']:.4f}"
        for pm_key in ['LPIPS', 'LPIPS_NO_ICP', 'SSIM', 'SSIM_NO_ICP', 'PSNR', 'PSNR_NO_ICP', 'COLOR_DIST', 'COLOR_DIST_NO_ICP']:
            if pm_key in means_all:
                row_data[pm_key] = f"{means_all[pm_key]:.4f}"

        return df_obj, row_data, means_all

    def get_metrics(self):
        data = []
        object_data = []
        overall_means = {'ADD': [], 'ADD-S': [], 'AR': [],
                         'VSD': [], 'MSSD': [], 'MSPD': [],
                         #'VSD_rec': [], 'MSSD_rec': [], 'MSPD_rec': [],
                         'R_error': [], 'T_error': []}
        if 'CHAMFER' in self.all_frames_metrics:
            overall_means['CHAMFER'] = []
        if 'CHAMFER_NO_ICP' in self.all_frames_metrics:
            overall_means['CHAMFER_NO_ICP'] = []
        if 'CHAMFER_ICP_WITH_SCALE' in self.all_frames_metrics:
            overall_means['CHAMFER_ICP_WITH_SCALE'] = []
        if 'CHAMFER_color' in self.all_frames_metrics:
            overall_means['CHAMFER_color'] = []
        if 'CHAMFER_color_NO_ICP' in self.all_frames_metrics:
            overall_means['CHAMFER_color_NO_ICP'] = []
        overall_means['chamfer_normalized'] = []
        overall_means['chamfer_normalized_icp_with_scale'] = []
        overall_means['ADDSS'] = []
        overall_means['ADDSS_10'] = []
        overall_means['ADDSS_05'] = []
        overall_means['ADDSS_02'] = []
        for pm_key in ['LPIPS', 'LPIPS_NO_ICP', 'SSIM', 'SSIM_NO_ICP', 'PSNR', 'PSNR_NO_ICP', 'COLOR_DIST', 'COLOR_DIST_NO_ICP']:
            overall_means[pm_key] = []

        for obj_id in self.metrics:

            df_obj, row_data, means_object = self.get_object_metrics(obj_id)

            data.append(row_data)

            object_data.append((obj_id, df_obj))

            overall_means['ADD'].append(means_object['ADD'])
            overall_means['ADD-S'].append(means_object['ADD-S'])
            overall_means['AR'].append(means_object['AR'])
            overall_means['VSD'].append(means_object['VSD'])
            overall_means['MSSD'].append(means_object['MSSD'])
            overall_means['MSPD'].append(means_object['MSPD'])
            overall_means['R_error'].append(means_object['R_error'])
            overall_means['T_error'].append(means_object['T_error'])
            if 'CHAMFER' in means_object:
                overall_means['CHAMFER'].append(means_object['CHAMFER'])
            if 'CHAMFER_NO_ICP' in means_object:
                overall_means['CHAMFER_NO_ICP'].append(means_object['CHAMFER_NO_ICP'])
            if 'CHAMFER_ICP_WITH_SCALE' in means_object:
                overall_means['CHAMFER_ICP_WITH_SCALE'].append(means_object['CHAMFER_ICP_WITH_SCALE'])
            if 'CHAMFER_color' in means_object:
                overall_means['CHAMFER_color'].append(means_object['CHAMFER_color'])
            if 'CHAMFER_color_NO_ICP' in means_object:
                overall_means['CHAMFER_color_NO_ICP'].append(means_object['CHAMFER_color_NO_ICP'])
            if 'chamfer_normalized' in means_object:
                overall_means['chamfer_normalized'].append(means_object['chamfer_normalized'])
            if 'chamfer_normalized_icp_with_scale' in means_object:
                overall_means['chamfer_normalized_icp_with_scale'].append(means_object['chamfer_normalized_icp_with_scale'])
            if 'ADDSS' in means_object:
                overall_means['ADDSS'].append(means_object['ADDSS'])
            if 'ADDSS_10' in means_object:
                overall_means['ADDSS_10'].append(means_object['ADDSS_10'])
            if 'ADDSS_05' in means_object:
                overall_means['ADDSS_05'].append(means_object['ADDSS_05'])
            if 'ADDSS_02' in means_object:
                overall_means['ADDSS_02'].append(means_object['ADDSS_02'])
            for pm_key in ['LPIPS', 'LPIPS_NO_ICP', 'SSIM', 'SSIM_NO_ICP', 'PSNR', 'PSNR_NO_ICP', 'COLOR_DIST', 'COLOR_DIST_NO_ICP']:
                if pm_key in means_object:
                    overall_means[pm_key].append(means_object[pm_key])

        overall_means = {k: np.mean(overall_means[k]) if overall_means[k] else np.nan for k in overall_means}

        total_samples = sum(len(self.metrics[obj_id]['ADD-S']) for obj_id in self.metrics) + self.total_failures

        mean_row = {
            'Class_ID': 'MEAN',
            'N_total': total_samples,
            'N_failed': self.total_failures,
            'ADD-S': f"{overall_means['ADD-S']:.1f}",
            'ADD': f"{overall_means['ADD']:.1f}",
            'AR': f"{overall_means['AR']:.1f}",
            'MSSD': f"{overall_means['MSSD']:.1f}",
            'MSPD': f"{overall_means['MSPD']:.1f}",
            'VSD': f"{overall_means['VSD']:.1f}",
        }
        if 'CHAMFER' in overall_means:
            mean_row['CHAMFER'] = f"{overall_means['CHAMFER']:.1f}"
        if 'CHAMFER_NO_ICP' in overall_means:
            mean_row['CHAMFER_NO_ICP'] = f"{overall_means['CHAMFER_NO_ICP']:.1f}"
        if 'CHAMFER_ICP_WITH_SCALE' in overall_means:
            mean_row['CHAMFER_ICP_WITH_SCALE'] = f"{overall_means['CHAMFER_ICP_WITH_SCALE']:.1f}"
        if 'CHAMFER_color' in overall_means:
            mean_row['CHAMFER_color'] = f"{overall_means['CHAMFER_color']:.1f}"
        if 'CHAMFER_color_NO_ICP' in overall_means:
            mean_row['CHAMFER_color_NO_ICP'] = f"{overall_means['CHAMFER_color_NO_ICP']:.1f}"
        if not np.isnan(overall_means.get('chamfer_normalized', np.nan)):
            mean_row['chamfer_normalized'] = f"{overall_means['chamfer_normalized']:.4f}"
        if not np.isnan(overall_means.get('chamfer_normalized_icp_with_scale', np.nan)):
            mean_row['chamfer_normalized_icp_with_scale'] = f"{overall_means['chamfer_normalized_icp_with_scale']:.4f}"
        if not np.isnan(overall_means.get('ADDSS', np.nan)):
            mean_row['ADDSS'] = f"{overall_means['ADDSS']:.4f}"
        if not np.isnan(overall_means.get('ADDSS_10', np.nan)):
            mean_row['ADDSS_10'] = f"{overall_means['ADDSS_10']:.4f}"
        if not np.isnan(overall_means.get('ADDSS_05', np.nan)):
            mean_row['ADDSS_05'] = f"{overall_means['ADDSS_05']:.4f}"
        if not np.isnan(overall_means.get('ADDSS_02', np.nan)):
            mean_row['ADDSS_02'] = f"{overall_means['ADDSS_02']:.4f}"
        for pm_key in ['LPIPS', 'LPIPS_NO_ICP', 'SSIM', 'SSIM_NO_ICP', 'PSNR', 'PSNR_NO_ICP', 'COLOR_DIST', 'COLOR_DIST_NO_ICP']:
            if not np.isnan(overall_means.get(pm_key, np.nan)):
                mean_row[pm_key] = f"{overall_means[pm_key]:.4f}"

        data.append(mean_row)

        df = pd.DataFrame(data)

        df_all_frames = pd.DataFrame(self.all_frames_metrics)
        means_all = {
            'Frame_ID': 'MEAN',
            'Class': 'ALL',
            'ADD-S': f"{df_all_frames['ADD-S'].mean() * 100:.1f}",
            'ADD': f"{df_all_frames['ADD'].mean() * 100:.1f}",
            'AR': f"{df_all_frames['AR'].mean() * 100:.1f}",
            'MSSD': f"{df_all_frames['MSSD'].mean() * 100:.1f}",
            'MSPD': f"{df_all_frames['MSPD'].mean() * 100:.1f}",
            'VSD': f"{df_all_frames['VSD'].mean() * 100:.1f}",
            'R_error': f"{df_all_frames['R_error'].mean():.1f}",
            'T_error': f"{df_all_frames['T_error'].mean():.1f}",
        }
        if 'CHAMFER' in df_all_frames:
            means_all['CHAMFER'] = f"{df_all_frames['CHAMFER'].mean():.1f}"
        if 'CHAMFER_NO_ICP' in df_all_frames:
            means_all['CHAMFER_NO_ICP'] = f"{df_all_frames['CHAMFER_NO_ICP'].mean():.1f}"
        if 'CHAMFER_ICP_WITH_SCALE' in df_all_frames:
            means_all['CHAMFER_ICP_WITH_SCALE'] = f"{df_all_frames['CHAMFER_ICP_WITH_SCALE'].mean():.1f}"
        if 'CHAMFER_color' in df_all_frames:
            means_all['CHAMFER_color'] = f"{df_all_frames['CHAMFER_color'].mean():.1f}"
        if 'CHAMFER_color_NO_ICP' in df_all_frames:
            means_all['CHAMFER_color_NO_ICP'] = f"{df_all_frames['CHAMFER_color_NO_ICP'].mean():.1f}"
        if 'chamfer_normalized' in df_all_frames:
            means_all['chamfer_normalized'] = f"{pd.to_numeric(df_all_frames['chamfer_normalized'], errors='coerce').mean():.4f}"
        if 'chamfer_normalized_icp_with_scale' in df_all_frames:
            means_all['chamfer_normalized_icp_with_scale'] = f"{pd.to_numeric(df_all_frames['chamfer_normalized_icp_with_scale'], errors='coerce').mean():.4f}"
        if 'ADDSS' in df_all_frames:
            means_all['ADDSS'] = f"{pd.to_numeric(df_all_frames['ADDSS'], errors='coerce').mean():.4f}"
        if 'ADDSS_10' in df_all_frames:
            means_all['ADDSS_10'] = f"{pd.to_numeric(df_all_frames['ADDSS_10'], errors='coerce').mean():.4f}"
        if 'ADDSS_05' in df_all_frames:
            means_all['ADDSS_05'] = f"{pd.to_numeric(df_all_frames['ADDSS_05'], errors='coerce').mean():.4f}"
        if 'ADDSS_02' in df_all_frames:
            means_all['ADDSS_02'] = f"{pd.to_numeric(df_all_frames['ADDSS_02'], errors='coerce').mean():.4f}"
        for pm_key in ['LPIPS', 'LPIPS_NO_ICP', 'SSIM', 'SSIM_NO_ICP', 'PSNR', 'PSNR_NO_ICP', 'COLOR_DIST', 'COLOR_DIST_NO_ICP']:
            if pm_key in df_all_frames:
                means_all[pm_key] = f"{pd.to_numeric(df_all_frames[pm_key], errors='coerce').mean():.4f}"
        df_all_frames = pd.concat([df_all_frames, pd.DataFrame([means_all])], ignore_index=True)

        return df, object_data, df_all_frames

    def save_metrics(self, output_path):
        df, object_data, df_all_frames = self.get_metrics()

        # create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Save as CSV
        df.to_csv(os.path.join(output_path, "0_mean_all_metrics_classes_results.csv" ), index=False)
        for obj_id, df_obj in object_data:
            df_obj.to_csv(os.path.join(output_path, f'{obj_id}_metrics_results.csv'), index=False)
        df_all_frames.to_csv(os.path.join(output_path, "0_all_frames_metrics_results.csv" ), index=False)

        # Save statistics (mean and median) for chamfer metrics
        # Use df_all_frames without the MEAN row for computing statistics
        df_numeric = df_all_frames[df_all_frames['Frame_ID'] != 'MEAN']
        stats_rows = []
        for stat_name, stat_fn in [('MEAN', 'mean'), ('MEDIAN', 'median')]:
            row = {'Statistic': stat_name}
            for col in ['ADD-S', 'CHAMFER', 'CHAMFER_NO_ICP', 'CHAMFER_ICP_WITH_SCALE', 'CHAMFER_color', 'CHAMFER_color_NO_ICP',
                        'chamfer_normalized', 'chamfer_normalized_icp_with_scale', 'ADDSS', 'ADDSS_10', 'ADDSS_05', 'ADDSS_02',
                        'LPIPS', 'LPIPS_NO_ICP', 'SSIM', 'SSIM_NO_ICP', 'PSNR', 'PSNR_NO_ICP',
                        'COLOR_DIST', 'COLOR_DIST_NO_ICP']:
                if col in df_numeric.columns:
                    values = pd.to_numeric(df_numeric[col], errors='coerce')
                    row[col] = f"{getattr(values, stat_fn)():.4f}"
            stats_rows.append(row)
        if stats_rows:
            df_stats = pd.DataFrame(stats_rows)
            df_stats.to_csv(os.path.join(output_path, "0_all_frames_metrics_results_statistics.csv"), index=False)

        # Save failure summary if there were any failures
        if self.total_failures > 0:
            failure_rows = []
            for obj_id in sorted(self.failures.keys()):
                for file_name in self.failures[obj_id]:
                    failure_rows.append({'Object_ID': obj_id, 'File_Name': file_name})
            df_failures = pd.DataFrame(failure_rows)
            df_failures.to_csv(os.path.join(output_path, "0_failed_samples.csv"), index=False)
            print(f"WARNING: {self.total_failures} samples failed and were counted as 0 for recall metrics (ADD-S). "
                  f"Chamfer metrics are computed over successful samples only. "
                  f"Details saved to 0_failed_samples.csv")

        print(f"Metrics saved to {output_path} (CSV format)")

    def compute_metrics_anchor(self, obj_id, pred_pose, pred_mesh, gt_pose, gt_mesh, gt_mesh_metadata):

        gt_diameter = gt_mesh_metadata['diameter']
        trans_disc = gt_mesh_metadata.get('symmetries_discrete', [])

        chamfer_dist, icp_transform = chamfer_distance_gt_mesh(gt_pose, gt_mesh, pred_pose, pred_mesh, use_icp=True, return_transform=True)
        chamfer_dist_no_icp = chamfer_distance_gt_mesh(gt_pose, gt_mesh, pred_pose, pred_mesh, use_icp=False)
        chamfer_dist_icp_with_scale, icp_transform_with_scale = chamfer_distance_gt_mesh(gt_pose, gt_mesh, pred_pose, pred_mesh, use_icp=True, with_scaling=True, return_transform=True)

        # Normalized Chamfer and ADDSS (gt_diameter is in meters, chamfer distances are in cm)
        diameter_cm = gt_diameter * 100.0
        chamfer_normalized = chamfer_dist / diameter_cm
        chamfer_normalized_icp_with_scale = chamfer_dist_icp_with_scale / diameter_cm
        addss = chamfer_dist_no_icp / diameter_cm
        addss_10 = float(addss < 0.10)
        addss_05 = float(addss < 0.05)
        addss_02 = float(addss < 0.02)

        if self.compute_color_chamfer:
            chamfer_dist_color, chamfer_dist_color_no_icp = chamfer_distance_gt_mesh_color(
                gt_pose, gt_mesh, pred_pose, pred_mesh, icp_transform=icp_transform
            )
        else:
            chamfer_dist_color = np.nan
            chamfer_dist_color_no_icp = np.nan

        #err_R, err_T = compute_RT_distances(pred_pose, gt_pose)
        #for r_th, t_th in self.pose_recall_th:
        #    succ_r, succ_t = err_R <= r_th, err_T <= t_th
        #    succ_pose = np.logical_and(succ_r, succ_t).astype(float)

        #add = compute_add_gen(pred_mesh.vertices, gt_mesh.vertices, pred_pose, gt_pose)
        adds = compute_adds_gen(pred_mesh.vertices, gt_mesh.vertices, pred_pose, gt_pose)

        #add_thres = float(add <= gt_diameter * 0.1)
        adds_thres = float(adds <= gt_diameter * 0.1)

        metrics = {
            "err_R": np.nan,
            "err_T": np.nan,
            "add": np.nan,
            "adds": adds,
            "add_thres": np.nan,
            "adds_thres": adds_thres,
            "mssd_err": np.nan,
            "mspd_err": np.nan,
            "vsd_errs": np.nan,
            "mean_mssd": np.nan,
            "mean_mspd": np.nan,
            "mssd_rec": np.nan,
            "mspd_rec": np.nan,
            "vsd_rec": np.nan,
            "mean_vsd": np.nan,
            "mean_ar": np.nan,
            "chamfer_dist": chamfer_dist,
            "chamfer_dist_no_icp": chamfer_dist_no_icp,
            "chamfer_dist_icp_with_scale": chamfer_dist_icp_with_scale,
            "chamfer_dist_color": chamfer_dist_color,
            "chamfer_dist_color_no_icp": chamfer_dist_color_no_icp,
            "chamfer_normalized": chamfer_normalized,
            "chamfer_normalized_icp_with_scale": chamfer_normalized_icp_with_scale,
            "addss": addss,
            "addss_10": addss_10,
            "addss_05": addss_05,
            "addss_02": addss_02,
            "icp_transform": icp_transform,
            "icp_transform_with_scale": icp_transform_with_scale,
        }

        return metrics

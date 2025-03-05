from dataclasses import dataclass

from contrastive_3d.datasets import monai_datalists, monai_transforms

@dataclass
class DatasetConfig:
    meta_data_path: str
    cache_dir: str
    image_data_path: str
    get_datalist: object
    transforms: object
    label_names: list = None
    num_classes: int = None

import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
import numpy as np
from copy import copy, deepcopy
from tqdm import tqdm

from contrastive_3d.utils import split_reports
from contrastive_3d.datasets import zero_shot_prompts
from contrastive_3d.datasets import zero_shot_utils


def get_stanford_datalist(split_type: str, fraction_train_data=None, config=None, dataset_config=None):
    REPORT_GENERATION = False
    SPLIT_INTO_5_PARTS = False

    data_frame = pd.read_csv(
        dataset_config.meta_data_path
    )
    data_frame['image_file'] = data_frame['image_file'].apply(lambda f: f.replace('/dataNAS/people/lblankem/abct_imaging_data/abct_compressed/', '/dataNAS/data/ct_data/abct_compressed/'))

    image_paths = glob.glob(
        os.path.join(dataset_config.image_data_path, "*.nii.gz")
    )
    image_paths_df = pd.DataFrame(image_paths, columns=["image_file"])
    # change all dataNAS to dataNASbak
    # data_frame["image_file"] = data_frame["image_file"].apply(lambda x: x.replace("dataNAS", "dataNASbak"))
    data_frame = pd.merge(data_frame, image_paths_df, on="image_file", how="inner")
    if split_type == "train":
        splits = [0, 1, 2]
    elif split_type == "val":
        splits = [3]
    elif split_type == "test":
        splits = [4]
    data_frame = data_frame[data_frame.split.isin(splits)]
    if split_type == "train":
        data_frame = data_frame.sample(frac=fraction_train_data, random_state=0).reset_index(drop=True)
    # elif split_type == "val":
    #     data_frame = data_frame.sample(frac=fraction_train_data, random_state=0).reset_index(drop=True)
    # elif split_type == "test":
    #     data_frame = data_frame.sample(frac=fraction_train_data, random_state=0).reset_index(drop=True)
    data_list = [
        {
            "image": row.image_file,
            "seg": row.seg_file,
            "labels": list(row[1:1693]),
            "findings": row.findings,
            "anon_accession": row.anon_accession,
        }
        for row in data_frame.itertuples(index=False)
    ]

    # data_list = []
    # for i in range(len(data_frame)):
    #     image_file_path = data_frame.iloc[i]["image_file"]
    #     findings = data_frame.iloc[i]["findings"]
    #     seg_file_path = data_frame.iloc[i]["seg_file"]
    #     labels = list(data_frame.iloc[i, 1:1693])
    #     # labels = list(data_frame.iloc[i, 3:1695])
    #     data_list.append(
    #         {
    #             "image": image_file_path,
    #             "seg": seg_file_path,
    #             "labels": labels,
    #             "findings": findings,
    #             "anon_accession": data_frame.iloc[i]["anon_accession"],
    #         }
    #     )
    parse_findings_list = False
    if not parse_findings_list:
        return data_list
    else:
        if REPORT_GENERATION or (split_type == "train"):
            # data_copy = deepcopy(data_list)
            # findings = [data_item["findings"] for data_item in data_copy]
            if SPLIT_INTO_5_PARTS:
                organ_system_names = ["lower thorax and liver", "solid organs", "bowel", "peritoneum and pelvic", "circulatory and musculoskeletal"]
            else:
                organ_system_names = [
                    "lower thorax",
                    "liver",
                    "gallbladder",
                    "spleen",
                    "pancreas",
                    "adrenal glands",
                    "kidneys",
                    "bowel",
                    "peritoneum",
                    "pelvic",
                    "circulatory",
                    "lymph nodes",
                    "musculoskeletal",
                ]

            # organ_system_datalists = {}
            # print("Mode: " + split_type)
            # for organ_system_name in organ_system_names:
            #     if REPORT_GENERATION:
            #         subsections, mask = split_reports.get_section(findings, probability_full_report = 0.0, normalize_prefix = True, organ_system_name = organ_system_name)
            #     else:
            #         subsections, mask = split_reports.get_section(findings, probability_full_report = 0.0, normalize_prefix = False, organ_system_name = organ_system_name)
            #     organ_system_datalists[organ_system_name] = [deepcopy(data_item) for i, data_item in enumerate(data_copy) if mask[i]]
            #     for i, _ in enumerate(organ_system_datalists[organ_system_name]):
            #         organ_system_datalists[organ_system_name][i]["findings"] = subsections[i]

            #     print("Number of parsed " + organ_system_name + ": " + str(len(organ_system_datalists[organ_system_name])))

            # # add all the data lists into self.data
            # # if mode == "train":
            # #     if REPORT_GENERATION:
            # #         data = []
            # #     for organ_system_name in self.organ_system_names:
            # if REPORT_GENERATION:
            #     data_sections = organ_system_datalists["lower thorax and liver"] + organ_system_datalists["solid organs"] + organ_system_datalists["bowel"] + organ_system_datalists["peritoneum and pelvic"] + organ_system_datalists["circulatory and musculoskeletal"]
            # else:
            #     data_sections = data_copy + organ_system_datalists["lower thorax and liver"] + organ_system_datalists["solid organs"] + organ_system_datalists["bowel"] + organ_system_datalists["peritoneum and pelvic"] + organ_system_datalists["circulatory and musculoskeletal"]
            # return data_sections

            new_data_list = []
            for data_item in tqdm(data_list, desc="Parsing stanford findings sections"):
                if REPORT_GENERATION:
                    data_item["findings_list"] = []
                else:
                    data_item["findings_list"] = [data_item["findings"]]
                for organ_system_name in organ_system_names:
                    if "findings_list" in data_item:
                        if REPORT_GENERATION:
                            section_text, mask = split_reports.get_section([data_item["findings"]], probability_full_report = 0.0, normalize_prefix = True, organ_system_name = organ_system_name)
                        else:
                            section_text, mask = split_reports.get_section([data_item["findings"]], probability_full_report = 0.0, normalize_prefix = False, organ_system_name = organ_system_name)
                        if mask[0]:
                            data_item["findings_list"].append(section_text[0])
                        else:
                            data_item["findings_list"].append("")
                if REPORT_GENERATION:
                    # if all the sections are empty, then remove the data item
                    if all([len(section) == 0 for section in data_item["findings_list"]]):
                        continue
                    else:
                        new_data_list.append(data_item)
                else:
                    new_data_list.append(data_item)

            data_list = new_data_list

            # print first 20 rows of the data list
            print("First 20 rows of train data list:")
            for i in range(20):
                print(data_list[i])
            return data_list
        else:
            return data_list


DATASETS = {
    "stanford": DatasetConfig(
        meta_data_path="/dataNAS/people/lblankem/contrastive-3d/data/stanford_labels.csv",
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/cache_dir",
        image_data_path="/dataNAS/data/ct_data/abct_compressed",
        get_datalist=get_stanford_datalist,
        transforms=monai_transforms.transforms_image,
    ),
    "stanford_report_generation": DatasetConfig(
        meta_data_path="/dataNAS/people/lblankem/contrastive-3d/data/stanford_labels_v1.csv",
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/cache_dir",
        image_data_path="/dataNAS/data/ct_data/abct_compressed",
        get_datalist=monai_datalists.get_stanford_report_generation_datalist,
        transforms=monai_transforms.transforms_image,
    ),
    "stanford_zero_shot_cls": DatasetConfig(
        meta_data_path="/dataNAS/people/akkumar/contrastive-3d/data/merged_labels_diseases.csv",
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/cache_dir",
        image_data_path="/dataNAS/data/ct_data/abct_compressed",
        get_datalist=monai_datalists.get_stanford_zero_shot_cls_datalist,
        transforms=monai_transforms.transforms_image,
    ),
    "stanford_ascites": DatasetConfig(
        meta_data_path="/dataNAS/people/lblankem/contrastive-3d/data/stanford_labels_ascites.csv",
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/cache_dir",
        image_data_path="/dataNAS/data/ct_data/abct_compressed",
        get_datalist=monai_datalists.get_stanford_ascites_datalist,
        transforms=monai_transforms.transforms_image,
    ),
    "stanford_pleural_effusion": DatasetConfig(
        meta_data_path="/dataNAS/people/lblankem/contrastive-3d/data/stanford_labels_pleural_effusion.csv",
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/cache_dir",
        image_data_path="/dataNAS/data/ct_data/abct_compressed",
        get_datalist=monai_datalists.get_stanford_pleural_effusion_datalist,
        transforms=monai_transforms.transforms_image,
    ),
    "stanford_renal_mass": DatasetConfig(
        meta_data_path="/dataNAS/people/lblankem/contrastive-3d/data/stanford_labels_renal_mass.csv",
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/cache_dir",
        image_data_path="/dataNAS/data/ct_data/abct_compressed",
        get_datalist=monai_datalists.get_stanford_renal_mass_datalist,
        transforms=monai_transforms.transforms_image,
    ),
    "stanford_hepatic_steatosis": DatasetConfig(
        meta_data_path="/dataNAS/people/lblankem/contrastive-3d/data/stanford_labels_hepatic_steatosis.csv",
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/cache_dir",
        image_data_path="/dataNAS/data/ct_data/abct_compressed",
        get_datalist=monai_datalists.get_stanford_hepatic_steatosis_datalist,
        transforms=monai_transforms.transforms_image,
    ),
    "stanford_appendicitis": DatasetConfig(
        meta_data_path="/dataNAS/people/lblankem/contrastive-3d/data/stanford_labels_appendicitis.csv",
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/cache_dir",
        image_data_path="/dataNAS/data/ct_data/abct_compressed",
        get_datalist=monai_datalists.get_stanford_appendicitis_datalist,
        transforms=monai_transforms.transforms_image,
    ),
    "stanford_gallstones": DatasetConfig(
        meta_data_path="/dataNAS/people/lblankem/contrastive-3d/data/stanford_labels_gallstones.csv",
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/cache_dir",
        image_data_path="/dataNAS/data/ct_data/abct_compressed",
        get_datalist=monai_datalists.get_stanford_gallstones_datalist,
        transforms=monai_transforms.transforms_image,
    ),
    "stanford_impressions": DatasetConfig(
        meta_data_path="/dataNAS/people/lblankem/contrastive-3d/data/stanford_labels.csv",
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/cache_dir",
        image_data_path="/dataNAS/data/ct_data/abct_compressed",
        get_datalist=monai_datalists.get_stanford_impressions_datalist,
        transforms=monai_transforms.transforms_image,
    ),
    "stanford_disease_prediction_all": DatasetConfig(
        meta_data_path="/dataNAS/people/lblankem/contrastive-3d/data/labels_adjusted_all_v1.csv",
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/cache_dir",
        image_data_path="/dataNAS/data/ct_data/abct_compressed",
        get_datalist=monai_datalists.get_stanford_disease_prediction_all_datalist,
        transforms=monai_transforms.transforms_image,
        label_names=["cvd", "ihd", "htn", "dm", "ckd", "ost"],
        num_classes=2,
    ),
    "stanford_ckd_disease_prediction": DatasetConfig(
        meta_data_path="/dataNAS/people/lblankem/contrastive-3d/data/labels_adjusted_all_v1.csv",
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/cache_dir",
        image_data_path="/dataNAS/data/ct_data/abct_compressed",
        get_datalist=monai_datalists.get_stanford_ckd_disease_prediction_datalist,
        transforms=monai_transforms.transforms_image,
        label_names=["ckd"],
        num_classes=2,
    ),
    "stanford_htn_disease_prediction": DatasetConfig(
        meta_data_path="/dataNAS/people/lblankem/contrastive-3d/data/labels_adjusted_all_v1.csv",
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/cache_dir",
        image_data_path="/dataNAS/data/ct_data/abct_compressed",
        get_datalist=monai_datalists.get_stanford_htn_disease_prediction_datalist,
        transforms=monai_transforms.transforms_image,
        label_names=["htn"],
        num_classes=2,
    ),
    "stanford_ihd_disease_prediction": DatasetConfig(
        meta_data_path="/dataNAS/people/lblankem/contrastive-3d/data/labels_adjusted_all_v1.csv",
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/cache_dir",
        image_data_path="/dataNAS/data/ct_data/abct_compressed",
        get_datalist=monai_datalists.get_stanford_ihd_disease_prediction_datalist,
        transforms=monai_transforms.transforms_image,
        label_names=["ihd"],
        num_classes=2,
    ),
    "stanford_dm_disease_prediction": DatasetConfig(
        meta_data_path="/dataNAS/people/lblankem/contrastive-3d/data/labels_adjusted_all_v1.csv",
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/cache_dir",
        image_data_path="/dataNAS/data/ct_data/abct_compressed",
        get_datalist=monai_datalists.get_stanford_dm_disease_prediction_datalist,
        transforms=monai_transforms.transforms_image,
        label_names=["dm"],
        num_classes=2,
    ),
    "stanford_ost_disease_prediction": DatasetConfig(
        meta_data_path="/dataNAS/people/lblankem/contrastive-3d/data/labels_adjusted_all_v1.csv",
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/cache_dir",
        image_data_path="/dataNAS/data/ct_data/abct_compressed",
        get_datalist=monai_datalists.get_stanford_ost_disease_prediction_datalist,
        transforms=monai_transforms.transforms_image,
        label_names=["ost"],
        num_classes=2,
    ),
    "stanford_ost_disease_prediction": DatasetConfig(
        meta_data_path="/dataNAS/people/lblankem/contrastive-3d/data/labels_adjusted_all_v1.csv",
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/cache_dir",
        image_data_path="/dataNAS/data/ct_data/abct_compressed",
        get_datalist=monai_datalists.get_stanford_cvd_disease_prediction_datalist,
        transforms=monai_transforms.transforms_image,
        label_names=["cvd"],
        num_classes=2,
    ),
    "ct_colonography": DatasetConfig(
        meta_data_path="/dataNAS/people/lblankem/contrastive-3d/data/colonography_labels.csv",
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/colonography_cache_dir",
        image_data_path="/dataNAS/people/lblankem/tcia/ct_colonography/nifti_images",
        get_datalist=monai_datalists.get_colonography_datalist,
        transforms=monai_transforms.transforms_image,
    ),
    "total_segmentator": DatasetConfig(
        meta_data_path=None,
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/total_segmentator_cache_dir",
        image_data_path="/dataNAS/people/lblankem/abct_imaging_data/total_segmentator",
        get_datalist=monai_datalists.get_total_segmentator_organ_datalist,
        transforms=monai_transforms.transforms_image,
    ),
    "total_segmentator_organs": DatasetConfig(
        meta_data_path=None,
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/total_segmentator_organs_cache_dir",
        image_data_path="/dataNAS/people/lblankem/abct_imaging_data/nnUNet_raw/Dataset000_TSORGANS",
        get_datalist=monai_datalists.get_total_segmentator_organ_datalist,
        transforms=monai_transforms.transforms_image_seg,
    ),
    "amos_organs": DatasetConfig(
        meta_data_path=None,
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/amos_organs_cache_dir",
        image_data_path="/dataNAS/people/lblankem/abct_imaging_data/amos22",
        get_datalist=monai_datalists.get_amos_organ_datalist,
        transforms=monai_transforms.transforms_image_seg,
    ),
    "oa_disease_diagnosis": DatasetConfig(
        meta_data_path=None,
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/oa_cache_dir",
        image_data_path="/dataNAS/people/aagatti/projects/OAI_Segmentation/oai_data_nii/",
        get_datalist=monai_datalists.get_oa_disease_diagnosis_datalist,
        transforms=monai_transforms.transforms_image_mr,
        label_names=["oa"],
        num_classes=1,
    ),
    "oa_disease_diagnosis_seg_as_image": DatasetConfig(
        meta_data_path=None,
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/oa_cache_dir",
        image_data_path="/dataNAS/people/aagatti/projects/OAI_Segmentation/oai_data_nii/",
        get_datalist=monai_datalists.get_oa_disease_diagnosis_seg_as_image_datalist,
        transforms=monai_transforms.transforms_image_mr,
        label_names=["oa"],
        num_classes=1,
    ),
    "oa_disease_prediction": DatasetConfig(
        meta_data_path=None,
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/oa_cache_dir",
        image_data_path="/dataNAS/people/aagatti/projects/OAI_Segmentation/oai_data_nii/",
        get_datalist=monai_datalists.get_oa_disease_prediction_datalist,
        transforms=monai_transforms.transforms_image_mr,
        label_names=["oa_incident_48"],
        num_classes=1,
    ),
    "oa_disease_prediction_seg_as_image": DatasetConfig(
        meta_data_path=None,
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/oa_cache_dir",
        image_data_path="/dataNAS/people/aagatti/projects/OAI_Segmentation/oai_data_nii/",
        get_datalist=monai_datalists.get_oa_disease_prediction_seg_as_image_datalist,
        transforms=monai_transforms.transforms_image_mr,
        label_names=["oa_incident_48"],
        num_classes=1,
    ),
    "oa_disease_staging": DatasetConfig(
        meta_data_path=None,
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/oa_cache_dir",
        image_data_path="/dataNAS/people/aagatti/projects/OAI_Segmentation/oai_data_nii/",
        get_datalist=monai_datalists.get_oa_disease_staging_datalist,
        transforms=monai_transforms.transforms_image_mr,
        label_names=["kl"],
        num_classes=4,
    ),
    "oa_disease_staging_seg_as_image": DatasetConfig(
        meta_data_path=None,
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/oa_cache_dir",
        image_data_path="/dataNAS/people/aagatti/projects/OAI_Segmentation/oai_data_nii/",
        get_datalist=monai_datalists.get_oa_disease_staging_seg_as_image_datalist,
        transforms=monai_transforms.transforms_image_mr,
        label_names=["kl"],
        num_classes=4,
    ),
    "oa_event_prediction": DatasetConfig(
        meta_data_path=None,
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/oa_cache_dir",
        image_data_path="/dataNAS/people/aagatti/projects/OAI_Segmentation/oai_data_nii/",
        get_datalist=monai_datalists.get_oa_event_prediction_datalist,
        transforms=monai_transforms.transforms_image_mr,
        label_names=["tkr_incident_108"],
        num_classes=1,
    ),
    "oa_event_prediction_seg_as_image": DatasetConfig(
        meta_data_path=None,
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/oa_cache_dir",
        image_data_path="/dataNAS/people/aagatti/projects/OAI_Segmentation/oai_data_nii/",
        get_datalist=monai_datalists.get_oa_event_prediction_seg_as_image_datalist,
        transforms=monai_transforms.transforms_image_mr,
        label_names=["tkr_incident_108"],
        num_classes=1,
    ),
    "oa_moaks_cartilage": DatasetConfig(
        meta_data_path=None,
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/oa_cache_dir",
        image_data_path="/dataNAS/people/aagatti/projects/OAI_Segmentation/oai_data_nii/",
        get_datalist=monai_datalists.get_oa_moaks_cartilage_datalist,
        transforms=monai_transforms.transforms_image_mr,
        label_names=[
            "cart_fem_ant_lat_thinning",
            "cart_fem_ant_med_thinning",
            "cart_fem_cent_lat_thinning",
            "cart_fem_cent_med_thinning",
            "cart_fem_post_lat_thinning",
            "cart_fem_post_med_thinning",
            "cart_fem_ant_lat_full_thickness",
            "cart_fem_ant_med_full_thickness",
            "cart_fem_cent_lat_full_thickness",
            "cart_fem_cent_med_full_thickness",
            "cart_fem_post_lat_full_thickness",
            "cart_fem_post_med_full_thickness",
        ],
        num_classes=1,
    ),
    "oa_moaks_cartilage_seg_as_image": DatasetConfig(
        meta_data_path=None,
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/oa_cache_dir",
        image_data_path="/dataNAS/people/aagatti/projects/OAI_Segmentation/oai_data_nii/",
        get_datalist=monai_datalists.get_oa_moaks_cartilage_seg_as_image_datalist,
        transforms=monai_transforms.transforms_image_mr,
        label_names=[
            "cart_fem_ant_lat_thinning",
            "cart_fem_ant_med_thinning",
            "cart_fem_cent_lat_thinning",
            "cart_fem_cent_med_thinning",
            "cart_fem_post_lat_thinning",
            "cart_fem_post_med_thinning",
            "cart_fem_ant_lat_full_thickness",
            "cart_fem_ant_med_full_thickness",
            "cart_fem_cent_lat_full_thickness",
            "cart_fem_cent_med_full_thickness",
            "cart_fem_post_lat_full_thickness",
            "cart_fem_post_med_full_thickness",
        ],
        num_classes=1,
    ),
    "oa_moaks_osteophytes": DatasetConfig(
        meta_data_path=None,
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/oa_cache_dir",
        image_data_path="/dataNAS/people/aagatti/projects/OAI_Segmentation/oai_data_nii/",
        get_datalist=monai_datalists.get_oa_moaks_osteophytes_datalist,
        transforms=monai_transforms.transforms_image_mr,
        label_names=[
            "osteophytes_fem_ant_lat_score",
            "osteophytes_fem_ant_med_score",
            "osteophytes_fem_cent_lat_score",
            "osteophytes_fem_cent_med_score",
            "osteophytes_fem_post_lat_score",
            "osteophytes_fem_post_med_score",
        ],
        num_classes=2,
    ),
    "oa_moaks_osteophytes_seg_as_image": DatasetConfig(
        meta_data_path=None,
        cache_dir="/dataNAS/people/lblankem/abct_imaging_data/oa_cache_dir",
        image_data_path="/dataNAS/people/aagatti/projects/OAI_Segmentation/oai_data_nii/",
        get_datalist=monai_datalists.get_oa_moaks_osteophytes_seg_as_image_datalist,
        transforms=monai_transforms.transforms_image_mr,
        label_names=[
            "osteophytes_fem_ant_lat_score",
            "osteophytes_fem_ant_med_score",
            "osteophytes_fem_cent_lat_score",
            "osteophytes_fem_cent_med_score",
            "osteophytes_fem_post_lat_score",
            "osteophytes_fem_post_med_score",
        ],
        num_classes=2,
    ),
    "ct_verse": DatasetConfig(
        meta_data_path="/dataNAS/people/akkumar/contrastive-3d/contrastive_3d/data/ct_verse.csv",
        cache_dir="/dataNAS/people/akkumar/contrastive-3d/abct_downsized/verse_cache_dir",
        image_data_path="/dataNAS/people/akkumar/contrastive-3d/contrastive_3d/data/ct_verse_extracted",
        get_datalist=monai_datalists.get_ct_verse,
        transforms=monai_transforms.transforms_image_verse,
        label_names=["vertebral_fracture"],
        num_classes=1,
    ),
}


def get_dataset_config(dataset_name):
    return DATASETS.get(dataset_name.lower())

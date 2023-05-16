import argparse

from segment_anything_tuning.eg_data_tools.eg_data_tools.annotation_processing.converters.convert_png_segmentation_to_coco import convert_supervisely_segmentation_to_coco

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help='Path to directory with classes directories')
    parser.add_argument("--result_coco_path", type=str, required=True)
    parser.add_argument("--num_process", type=int, default=4, required=False)
    parser.add_argument("--class_names", type=str, nargs="+", required=False, )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    
    convert_supervisely_segmentation_to_coco(
        png_img_folder=args.input_dir,
        coco_ann_path=args.result_coco_path,
        processes_number=args.num_process,
        class_names=args.class_names,
    )



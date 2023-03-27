import h5py
import numpy as np
import argparse

def convert_tabletop(fp):
    f = h5py.File(fp, 'r+')
    for vid_idx in range(f['data'].shape[0]):
        for frame_idx in range(f['data'].shape[1]):
            mask = f['mask'][vid_idx, frame_idx]
            unique_idx = np.unique(mask)
            unique_idx = np.delete(unique_idx, 0)
            unique_idx = np.delete(unique_idx, -1)
            mask[np.logical_and(mask != 0, mask != 65535)] += len(unique_idx)
            for actual_idx, cur_idx in enumerate(unique_idx):
                mask[mask == cur_idx+len(unique_idx)] = unique_idx[-actual_idx-1]

            f['mask'][vid_idx, frame_idx, ...] = mask
    
    f.close()

def convert_binidx(fp):
    f = h5py.File(fp, 'r+')
    for vid_idx in range(f['data'].shape[0]):
        for frame_idx in range(f['data'].shape[1]):
            seg = f['mask'][vid_idx, frame_idx]
            seg[seg!=65535] += 1
            bin_id = np.unique(seg)[-2]
            seg[seg==bin_id] = 0

            f['mask'][vid_idx, frame_idx, ...] = seg 
    
    f.close()

def convert_bin(fp, output_fp):
    source_f = h5py.File(fp, 'r')
    target_f = h5py.File(output_fp, 'w')
    frame0_data = source_f['frame0_data'][:]
    frame0_mask = source_f['frame0_mask'][:]
    frame0_depth = source_f['frame0_depth'][:]
    frame0_metadata = source_f['frame0_metadata'][:]
    frame1_data = source_f['frame1_data'][:]
    frame1_mask = source_f['frame1_mask'][:]
    frame1_depth = source_f['frame1_depth'][:]
    frame1_metadata = source_f['frame1_metadata'][:]
    
    combined_data = np.concatenate((np.expand_dims(frame0_data,axis=1), np.expand_dims(frame1_data,axis=1)), axis=1)
    combined_mask = np.concatenate((np.expand_dims(frame0_mask,axis=1), np.expand_dims(frame1_mask,axis=1)), axis=1)
    combined_depth = np.concatenate((np.expand_dims(frame0_depth,axis=1), np.expand_dims(frame1_depth,axis=1)), axis=1)
    combined_metadata = np.concatenate((np.expand_dims(frame0_metadata,axis=1), np.expand_dims(frame1_metadata,axis=1)), axis=1)
    target_f.create_dataset('data', data=combined_data)
    target_f.create_dataset('mask', data=combined_mask)
    target_f.create_dataset('depth', data=combined_depth)
    target_f.create_dataset('metadata', data=combined_metadata)

    source_f.close()
    target_f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None, help='Path to input file')
    parser.add_argument('--output', type=str, default=None, help='Path to output file')
    parser.add_argument('--convert_tabletop', action='store_true', help='Convert tabletop dataset')
    parser.add_argument('--handle_bin_idx', action='store_true', help='Handle bin index')

    args = parser.parse_args()
    if args.convert_tabletop:
        convert_tabletop(args.input)
    else:
        if args.output is None:
            if args.handle_bin_idx:
                convert_binidx(args.input)
        else:
            convert_bin(args.input, args.output)
            if args.handle_bin_idx:
                convert_binidx(args.output)
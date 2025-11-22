#!/usr/bin/env python3
import sys
import numpy as np
from PIL import Image

def convert_image_to_header(image_path, output_path='src/img.h'):
    try:

        print(f"Loading image: {image_path}")
        img = Image.open(image_path).convert('RGB')
        print(f"Original size: {img.size}")

        img = img.resize((160, 160))
        print(f"Resized to: 160x160")

        img_array = np.array(img, dtype=np.uint8)

        img_flat = img_array.flatten()

        print(f"Total values: {len(img_flat)} (should be 76800)")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('#ifndef __IMG_H__\n')
            f.write('#define __IMG_H__\n\n')
            f.write('const unsigned char img_data[] = {\n')

            for i, val in enumerate(img_flat):
                if i % 16 == 0:
                    f.write('  ')
                f.write(f'{val}')
                if i < len(img_flat) - 1:
                    f.write(', ')
                if (i + 1) % 16 == 0:
                    f.write('\n')

            if len(img_flat) % 12 != 0:
                f.write('\n')

            f.write('};\n\n')
            f.write('#endif // __IMG_H__\n')

    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)

    input_image = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'src/img.h'

    convert_image_to_header(input_image, output_file)

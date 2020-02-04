from PIL import Image

# img = Image.open('../../colon_seg/data/marchantia_imgs/g2_t017_c001.png')
# gt_img = Image.open('../../colon_seg/data/marchantia_gt/invert_g2_t017_c001.png')
ins_img = Image.open('../../instance_segmentation/marchantia_data/marchantia_ins_gt/g2_t002_c001.png')
sem_img = Image.open('../../instance_segmentation/marchantia_data/marchantia_sem_gt/invert_g2_t002_c001.png')

assert ins_img.size == sem_img.size

(imwidth, imheight) = sem_img.size

imsize = 128

for i in range((int)(imwidth / imsize)):
    for j in range((int)(imwidth / imsize)):
        # img_cropped = img.crop((i*imsize, j*imsize, i*imsize + imsize, j*imsize + imsize))
        # gt_img_cropped = gt_img.crop((i*imsize, j*imsize, i*imsize + imsize, j*imsize + imsize))
        ins_img_cropped = ins_img.crop((i*imsize, j*imsize, i*imsize + imsize, j*imsize + imsize))
        sem_img_cropped = sem_img.crop((i*imsize, j*imsize, i*imsize + imsize, j*imsize + imsize))
        color_extrema = sem_img_cropped.convert("L").getextrema()
        if not (color_extrema[0] == color_extrema[1]):
                # img_cropped.save('../../colon_seg/data/cropped_imgs/17_{:2d}_cropped.png'.format(i + j*8))
                # gt_img_cropped.save('../../colon_seg/data/inverted_gt/17_{:2d}_cropped.png'.format(i + j*8))
                ins_img_cropped.save('../../instance_segmentation/marchantia_data/all_cropped_gt/2_{:2d}_cropped.png'.format(i + j*8))

        


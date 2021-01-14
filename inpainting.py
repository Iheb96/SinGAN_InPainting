import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='training image name', required=True)
    parser.add_argument('--ref_dir', help='input reference dir', default='Input/Inpainting')
    parser.add_argument('--ref_name', help='reference image name', required=True)
    parser.add_argument('--inpainting_start_scale', help='inpainting injection scale', type=int, required=True)
    parser.add_argument('--mode', help='task to be done', default='inpainting')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
    #elif (os.path.exists(dir2save)):
    #    print("output already exist")
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        real = functions.adjust_scales2image(real, opt)
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
        if (opt.inpainting_start_scale < 1) | (opt.inpainting_start_scale > (len(Gs)-1)):
            print("injection scale should be between 1 and %d" % (len(Gs)-1))
        else:
            #Filling the part to be inpainted with the average of its surrounding pixels
            im_mask = cv2.imread('%s/%s_mask%s' % (opt.ref_dir, opt.ref_name[:-4], opt.ref_name[-4:]))
            img = cv2.imread('%s/%s' % (opt.input_dir, opt.ref_name))
            im_mask = im_mask / 255
            for k in range(3):
                img[:, :, k][im_mask[:, :, k] == 1] = img[:, :, k][im_mask[:, :, k] == 0].mean()

            cv2.imwrite('%s/%s_global_mean%s' % (opt.input_dir, opt.ref_name[:-4], opt.ref_name[-4:]), img)
            ref = functions.read_image_dir('%s/%s_global_mean%s' % (opt.input_dir, opt.ref_name[:-4], opt.ref_name[-4:]), opt)
            mask = functions.read_image_dir('%s/%s_mask%s' % (opt.ref_dir,opt.ref_name[:-4],opt.ref_name[-4:]), opt)
            if ref.shape[3] != real.shape[3]:
                mask = imresize_to_shape(mask, [real.shape[2], real.shape[3]], opt)
                mask = mask[:, :, :real.shape[2], :real.shape[3]]
                ref = imresize_to_shape(ref, [real.shape[2], real.shape[3]], opt)
                ref = ref[:, :, :real.shape[2], :real.shape[3]]
            mask = functions.dilate_mask(mask, opt)

            N = len(reals) - 1
            n = opt.inpainting_start_scale
            in_s = imresize(ref, pow(opt.scale_factor, (N - n + 1)), opt)
            in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
            in_s = imresize(in_s, 1 / opt.scale_factor, opt)
            in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            out = SinGAN_generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            out = (1-mask)*real+mask*out
            plt.imsave('%s/start_scale=%d.png' % (dir2save,opt.inpainting_start_scale), functions.convert_image_np(out.detach()), vmin=0, vmax=1)

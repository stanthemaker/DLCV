# with torch.no_grad():
#     # *********************
#     # *    report      *
#     # *********************
#     f_imgs_sample = (generator(sample_noise).data + 1) / 2.0
#     filename = os.path.join(
#         "/home/stan/hw2-stanthemaker/problem1_GAN/sample_output",
#         f"{train_name}_32.png",
#     )
#     vutils.save_image(f_imgs_sample, filename, nrow=8)

#     imgs = generator(noise).detach().cpu()
#     for i in range(n_outputs):
#         vutils.save_image(imgs[i], os.path.join(output_path, f"{i}.png"))

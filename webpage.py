from jinja2 import Environment, FileSystemLoader

def wrtie_webpage(infile, outfile, data, KS_values, images):
    env = Environment(loader=FileSystemLoader('./'))

    template = env.get_template(infile)

    with open(outfile, 'w+', encoding='utf-8') as f:
        out = template.render(
            items = data,
            KS_table = images['KS_value'],
            KS_value = KS_values,
            BDT_log = images['BDT_log'],
            BDT = images['BDT'],
            Folding_log = images['Folding_log'],
            Folding = images['Folding'],
            epoch_loss = images['epoch_loss'],
            GAN_corre = images['GAN_corre'],
            DSS_corre = images['DSS_corre'],
            D_network = images['D_network'],
            G_network = images['G_network']
        )
        f.write(out)
        f.close()

if __name__ == '__main__':
    test_data = {'in_file': '/lustre/collider/hezhengting/TrainningData/PN_ECAL_ana_4e5.hdf5', 'disc_lr': 0.001,
                 'end_learning_rate': 0.0001, 'disc_opt': 'Adam', 'adam_beta_1': 0.5, 'adam_beta_2': 0.9,
                 'decay_steps': 1000, 'decay_power': 2, 'decay_rate': 0.9, 'gen_lr': 0.0005, 'gen_opt': 'Nadam',
                 'energy_cut': 0.001, 'generator_extra_step': 1, 'discriminator_extra_steps': 1, 'batch_size': 600,
                 'final_layer_activation': 'softplus', 'z_alpha': 0.9, 'z_beta': 0.1, 'g_network_type': 'DownSampling',
                 'use_latent_optimization': True, 'lambda_E': 100.0, 'E_factor': 0, 'latent_size': 1024}
    infile = 'template.html'
    outfile = 'index.html'
    image_path = 'images/'
    images = {
        'KS_value' : image_path + 'KS_value.png',
        'BDT' : image_path + 'BDT.png',
        'BDT_log': image_path + 'BDT_log.png',
        'Folding': image_path + 'Folding.png',
        'Folding_log' : image_path + 'Folding_log.png',
        'epoch_loss' : image_path + 'epoch_loss.png',
        'GAN_corre' : image_path + 'GAN_corre.png',
        'DSS_corre' : image_path + 'DSS_corre.png',
        'D_network' : image_path + 'D_network.png',
        'G_network': image_path + 'G_network.png',
    }
    KS_values = {
        'original' : 0.382222,
        'Gradient Boosted Reweighter' : 0.068429,
        'Bins - based: total energy' : 0.347552,
        'Bins - based: total energy and x_sq' : 0.242355,
        'Folding reweighter': 0.160500,
    }

    wrtie_webpage(infile, outfile, test_data, KS_values, images)
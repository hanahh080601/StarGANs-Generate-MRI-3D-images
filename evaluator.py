import os 
import torch 
from data_loader import get_loader

class Evaluator(object):
    """Evaluator for StarGAN."""

    def __init__(self, config):
        """Initialize configurations."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = config.mode
        self.c_dim = config.c_dim
        self.image_size = config.image_size
        self.batch_size = config.batch_size
        self.test_iters = config.test_iters
        self.model_save_dir = config.model_save_dir
        self.input_contrast = config.input_contrast
        self.test_image_dir = config.brats2020_image_dir if config.dataset == 'BraTS2020' else config.ixi_image_dir


    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out
        
    def create_labels(self, c_org, c_dim=4):
        """Generate target domain labels for debugging and testing."""
        c_trg_list = []
        for i in range(c_dim):
            c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)
            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def evaluate(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        data_loader = get_loader(os.path.join(self.test_image_dir, self.input_contrast), self.image_size, self.batch_size, self.mode)
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))

    def evaluate_multi(self):
        """Translate images using StarGAN trained on multiple datasets."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(self.brats2020_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_brats2020_list = self.create_labels(c_org, self.c_dim)
                c_ixi_list = self.create_labels(c_org, self.c2_dim)
                zero_brats2020 = torch.zeros(x_real.size(0), self.c_dim).to(self.device)            # Zero vector for BraTS2020.
                zero_ixi = torch.zeros(x_real.size(0), self.c2_dim).to(self.device)             # Zero vector for IXI.
                mask_brats2020 = self.label2onehot(torch.zeros(x_real.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
                mask_ixi = self.label2onehot(torch.ones(x_real.size(0)), 2).to(self.device)     # Mask vector: [0, 1].

                # Translate images.
                x_fake_list = [x_real]
                for c_brats2020 in c_brats2020_list:
                    c_trg = torch.cat([c_brats2020, zero_ixi, mask_brats2020], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))
                for c_ixi in c_ixi_list:
                    c_trg = torch.cat([zero_brats2020, c_ixi, mask_ixi], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))
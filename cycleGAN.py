import torch
from model_loader import getG, getD
from losses import vanillaGANLoss, LSGANLoss, IdentityLoss, CycleLoss
from data_buffer import ReplayBuffer
import wandb

def G_output2numpy(output):
    output = output.cpu().numpy().transpose((1, 2, 0))
    return output


class CycleGANModel():

    def __init__(self, opt):

        # device
        self.device = torch.device('cuda') if torch.cuda.is_available() and opt.use_cuda else torch.device('cpu')

        # G and D
        self.G_ab = getG(opt).to(self.device)
        self.G_ba = getG(opt).to(self.device)
        self.D_a = getD(opt).to(self.device)
        self.D_b = getD(opt).to(self.device)

        # data replay buffer
        self.buffer_A = ReplayBuffer() # buffer size 50
        self.buffer_B = ReplayBuffer()

        # loss functions
        if opt.G_mode == "vanilla":
            self.ganloss = vanillaGANLoss()
        elif opt.G_mode == "lsgan":
            self.ganloss = LSGANLoss()
        else:
            raise NotImplementedError('check G loss type')

        self.identityloss = IdentityLoss()
        self.lambda_id = opt.lambda_id

        self.cycleloss = CycleLoss()
        self.lambda_cycle = opt.lambda_cycle

        # optimizers
        self.optimizer_G_ab = torch.optim.Adam(self.G_ab.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.optimizer_G_ba = torch.optim.Adam(self.G_ba.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.optimizer_D_a = torch.optim.Adam(self.D_a.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.optimizer_D_b = torch.optim.Adam(self.D_b.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

        # learning schedulers

        # wandb
        self.use_wandb = opt.use_wandb
        if self.use_wandb:
            wandb.init(project=opt.wandb_project)
            wandb.config.update(opt)

            wandb.watch(self.G_ab)
            wandb.watch(self.G_ba)
            wandb.watch(self.D_a)
            wandb.watch(self.D_b)

    def train_step(self, A, B, valid=False):

        A = A.to(self.device)
        B = B.to(self.device)

        # G ab
        self.optimizer_G_ab.zero_grad()
        self.optimizer_G_ba.zero_grad()

        ## Identity loss
        loss_id_A = self.identityloss(self.G_ba(A), A)
        loss_id_B = self.identityloss(self.G_ab(B), B)

        loss_identity = (loss_id_A + loss_id_B)

        ## GAN loss
        fake_B = self.G_ab(A)
        fake_A = self.G_ba(B)

        logit_fake_B = self.D_b(fake_B)
        logit_fake_A = self.D_a(fake_A)

        ones = torch.ones_like(logit_fake_B).to(self.device)
        zeros = torch.zeros_like(logit_fake_B).to(self.device)

        loss_gan_ab = self.ganloss(logit_fake_B, ones)
        loss_gan_ba = self.ganloss(logit_fake_A, ones)

        loss_gan = (loss_gan_ab + loss_gan_ba)

        ## Cycle loss
        recon_A = self.G_ba(fake_B)
        recon_B = self.G_ab(fake_A)

        loss_cycle_A = self.cycleloss(recon_A, A)
        loss_cycle_B = self.cycleloss(recon_B, B)

        loss_cycle = (loss_cycle_A + loss_cycle_B)

        ## total G loss
        loss_G = loss_gan + self.lambda_id * loss_identity + self.lambda_cycle * loss_cycle

        if not valid:
            loss_G.backward()
            self.optimizer_G_ab.step()
            self.optimizer_G_ba.step()

        # D train

        # D_a
        self.optimizer_D_a.zero_grad()

        fake_A = self.buffer_A.push_and_pop(fake_A).to(self.device)

        loss_D_a_real = self.ganloss(self.D_a(A), ones)
        loss_D_a_fake = self.ganloss(self.D_a(fake_A.detach()), zeros)

        loss_Da = (loss_D_a_real + loss_D_a_fake)

        if not valid:
            loss_Da.backward()
            self.optimizer_D_a.step()

        # D_b
        self.optimizer_D_b.zero_grad()

        fake_B = self.buffer_B.push_and_pop(fake_B).to(self.device)

        loss_D_b_real = self.ganloss(self.D_b(B), ones)
        loss_D_b_fake = self.ganloss(self.D_b(fake_B.detach()), zeros)

        loss_Db = (loss_D_b_real + loss_D_b_fake)

        if not valid:
            loss_Db.backward()
            self.optimizer_D_b.step()

        return loss_G.item(), loss_gan.item(), loss_cycle.item(), loss_identity.item(), loss_Da.item(), loss_Db.item()

    def train(self, opt, train_loader, valid_loader=None):

        for epoch in range(opt.epochs):

            # train
            loss_Gs = 0
            loss_gans = 0
            loss_cycles = 0
            loss_identitys = 0
            loss_Das = 0
            loss_Dbs = 0

            for idx, (A, B) in enumerate(train_loader):
                loss_G, loss_gan, loss_cycle, loss_identity, loss_Da, loss_Db = self.train_step(A, B)

                loss_Gs += loss_G
                loss_gans += loss_gan
                loss_cycles += loss_cycle
                loss_identitys += loss_identity
                loss_Das += loss_Da
                loss_Dbs += loss_Db

                if idx % 100 == 0 or idx == len(train_loader) - 1:
                    print('Epoch {:4} : [{:5}/{:5}] : G:{:6.06} \tDa:{:6.06} \tDb:{:6.06}'.format(
                        epoch, (idx + 1) * opt.batch_size, len(train_loader) * opt.batch_size, loss_Gs / (idx + 1),
                               loss_Das / (idx + 1), loss_Dbs / (idx + 1)))

            if self.use_wandb:
                wandb.log({"loss_G": loss_Gs / (idx + 1),
                           "loss_gan": loss_gans / (idx + 1),
                           "loss_cycle": loss_cycles / (idx + 1),
                           "loss_identity": loss_identitys / (idx + 1),
                           "loss_Da": loss_Das / (idx + 1),
                           "loss_Db": loss_Dbs / (idx + 1),
                           })

            # validation
            if valid_loader:
                valid_loss_Gs = 0
                valid_loss_gans = 0
                valid_loss_cycles = 0
                valid_loss_identitys = 0
                valid_loss_Das = 0
                valid_loss_Dbs = 0
                with torch.no_grad():
                    for idx, (valid_A, valid_B) in enumerate(valid_loader):
                        valid_loss_G, valid_loss_gan, valid_loss_cycle, valid_loss_identity, valid_loss_Da, valid_loss_Db = self.train_step(
                            valid_A, valid_B, valid=True)

                        valid_loss_Gs += valid_loss_G
                        valid_loss_gans += valid_loss_gan
                        valid_loss_cycles += valid_loss_cycle
                        valid_loss_identitys += valid_loss_identity
                        valid_loss_Das += valid_loss_Da
                        valid_loss_Dbs += valid_loss_Db

                    print(' >> Valid  : [{:5}/{:5}] : G:{:06.6} \tDa:{:06.6} \tDb:{:06.6}'.format(
                        (idx + 1) * opt.batch_size, len(valid_loader) * opt.batch_size, valid_loss_Gs / (idx + 1),
                        valid_loss_Das / (idx + 1), valid_loss_Dbs / (idx + 1)))

                    if self.use_wandb:
                        wandb.log({"valid_loss_G": valid_loss_Gs / (idx + 1),
                                   "valid_loss_gan": valid_loss_gans / (idx + 1),
                                   "valid_loss_cycle": valid_loss_cycles / (idx + 1),
                                   "valid_loss_identity": valid_loss_identitys / (idx + 1),
                                   "valid_loss_Da": valid_loss_Das / (idx + 1),
                                   "valid_loss_Db": valid_loss_Dbs / (idx + 1),
                                   })
            # model save            
            torch.save(self, opt.save_path + '/{}.pth'.format(epoch))

            # image log
            if self.use_wandb:
            	wandb.save(opt.save_path + '/{}.pth'.format(epoch))
                
                with torch.no_grad():
                    id_A = self.G_ba(valid_A.to(self.device))
                    id_B = self.G_ab(valid_B.to(self.device))
                    fake_A = self.G_ba(valid_B.to(self.device))
                    fake_B = self.G_ab(valid_A.to(self.device))
                    recon_A = self.G_ba(fake_B.to(self.device))
                    recon_B = self.G_ab(fake_A.to(self.device))

                    wandb.log({"id_A": wandb.Image(G_output2numpy(id_A[0])),
                               "id_B": wandb.Image(G_output2numpy(id_B[0])),
                               "fake_A": wandb.Image(G_output2numpy(fake_A[0])),
                               "fake_B": wandb.Image(G_output2numpy(fake_B[0])),
                               "recon_A": wandb.Image(G_output2numpy(recon_A[0])),
                               "recon_B": wandb.Image(G_output2numpy(recon_B[0])),
                               })


        #     


if __name__ == "__main__":
    from config import get_arguments

    # import matplotlib.pyplot as plt
    parser = get_arguments()
    opt = parser.parse_args()
    cyclegan = CycleGANModel(opt)
    A = torch.randn(3, 3, 52, 52)
    B = torch.randn(3, 3, 52, 52)

    cyclegan.train_step(A, B)

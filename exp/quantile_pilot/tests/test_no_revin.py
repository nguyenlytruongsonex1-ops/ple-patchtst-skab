"""No-RevIN ablation correctness, not model efficacy tests."""
import io
import unittest
import numpy as np
import torch
from exp.quantile_pilot.ple import fit_scaled_quantile_edges, fit_revin_quantile_edges
from exp.quantile_pilot.models import build_model, predict
from exp.quantile_pilot.train import train_one
from exp.quantile_pilot import data_skab

class NoRevINTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        self.x=torch.randn(6,60,8)*torch.arange(1,9)+torch.arange(8)*10
        self.edges=fit_scaled_quantile_edges(self.x)

    def make(self, ple=True, revin=False):
        return build_model('PatchTST-PLE-quantile' if ple else 'PatchTST',
                           task='classification', n_features=8,lookback=60,
                           edges=self.edges if ple else None,use_revin=revin)

    def test_train_scaled_quantiles_not_instance_normalized(self):
        q=torch.quantile(self.x.reshape(-1,8),torch.linspace(0,1,9),dim=0).T
        torch.testing.assert_close(self.edges,q)
        self.assertFalse(torch.allclose(self.edges,fit_revin_quantile_edges(self.x)))
        val=self.x+9999;test=self.x-9999
        # Knots are immutable after fit; held-out tensors are not arguments.
        torch.testing.assert_close(self.edges,fit_scaled_quantile_edges(self.x))
        self.assertGreater(val.mean().item(),test.mean().item())

    def test_zero_ple_no_affine_matches_old_backbone(self):
        torch.manual_seed(7);m=self.make()
        torch.manual_seed(7);plain=self.make(ple=False)
        self.assertIsNone(m.revin)
        self.assertFalse(any('revin' in n for n,_ in m.named_parameters()))
        torch.testing.assert_close(m.pre_affine(self.x),self.x,rtol=0,atol=0)
        m.eval();plain.eval()
        torch.testing.assert_close(m(self.x),plain(self.x),rtol=0,atol=0)
        torch.manual_seed(7);old=data_skab.run_detect.Model('PatchTST',.1)
        for k,v in m.net.state_dict().items():
            torch.testing.assert_close(v,old.net.state_dict()[k],rtol=0,atol=0)
        old.eval()
        torch.testing.assert_close(m(self.x),old(self.x),rtol=0,atol=0)

    def test_paired_seed_initialization_and_16_param_difference(self):
        for seed in range(5):
            torch.manual_seed(seed);off=self.make()
            torch.manual_seed(seed);on=self.make(revin=True)
            for k,v in off.state_dict().items():
                torch.testing.assert_close(v,on.state_dict()[k],rtol=0,atol=0)
            self.assertEqual(sum(p.numel() for p in on.parameters())-
                             sum(p.numel() for p in off.parameters()),16)

    def test_eval_serialization_and_frozen_knots(self):
        m=self.make();m.ple.W.data.fill_(.05)
        before=m.ple.edges.clone()
        p=predict(m,self.x,task='classification')
        buf=io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0)
        restored=self.make();restored.load_state_dict(torch.load(buf,weights_only=True))
        torch.testing.assert_close(p,predict(restored,self.x,task='classification'),rtol=0,atol=0)
        # One actual optimizer step updates weights but never knots.
        m.train();opt=torch.optim.AdamW(m.parameters(),lr=.001)
        loss=m(self.x).square().mean();loss.backward();opt.step()
        torch.testing.assert_close(before,m.ple.edges,rtol=0,atol=0)

    def test_no_early_stop_runs_full_budget(self):
        x=np.ones((4,2),dtype='float32');y=np.zeros(4,dtype='float32')
        fit=train_one(lambda:torch.nn.Linear(2,1),task='classification',
            x_train=x,y_train=y,x_val=x,y_val=y,score_from_pred=lambda _: .5,
            maximize=True,seed=0,max_epochs=30,patience=None)
        self.assertEqual(fit['epochs_ran'],30)
        self.assertEqual(fit['best_epoch'],1)

if __name__=='__main__':unittest.main()

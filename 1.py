self.model.eval()
        pred_ = []
        truth_ = []
        loss = 0.0
        with torch.no_grad():
            cn = 0
            for data in testset:
                data = data.to(self.args.device, non_blocking=pin_memory)
                pred = self.model(data, self.args.adj)
                loss += func.mse_loss(data.y, pred, reduction="mean")
                pred, _ = to_dense_batch(pred, batch=data.batch)
                data.y, _ = to_dense_batch(data.y, batch=data.batch)
                pred_.append(pred.cpu().data.numpy())
                truth_.append(data.y.cpu().data.numpy())
                cn += 1
            loss = loss/cn
            self.args.logger.info("[*] loss:{:.4f}".format(loss))
            pred_ = np.concatenate(pred_, 0)
            truth_ = np.concatenate(truth_, 0)
            mae = base_framework.metric(truth_, pred_, self.args)
            return loss
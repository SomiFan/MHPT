import time
import datetime
import torch
import torch.nn.functional as F
import numpy as np
from skimage import io
from timm.utils import AverageMeter
from utils import get_grad_norm, avg_metric, sam, ergas, bgr2rgb, qnr


def train_one_epoch(
        config,
        model,
        criterion,
        data_loader,
        optimizer,
        epoch,
        lr_scheduler,
        logger,
        writer_dict,
        device
):
    writer = writer_dict['writer']
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    if not config.TRAIN.SING_LOSS:
        if config.TRAIN.L1:
            l1_meter = AverageMeter()
        if config.TRAIN.L2:
            l2_meter = AverageMeter()
        if config.TRAIN.SPCSIM:
            spcsim_meter = AverageMeter()
        if config.TRAIN.BR:
            br_meter = AverageMeter()
        if config.TRAIN.FAMSIS:
            famsis_meter = AverageMeter()
        if config.TRAIN.PSMSIS:
            psmsis_meter = AverageMeter()
        if config.TRAIN.FAMEDGE:
            famedge_meter = AverageMeter()
        if config.TRAIN.PSMEDGE:
            psmedge_meter = AverageMeter()
        if config.TRAIN.GRADLOSS:
            gradloss_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, imgs in enumerate(data_loader):
        global_steps = writer_dict['train_global_steps']
        # mspan, target = imgs[0].view(-1, config.MODEL.NUM_MS_BANDS + 1, config.MODEL.PAN_SIZE, config.MODEL.PAN_SIZE), imgs[1].view(-1, config.MODEL.NUM_MS_BANDS, config.MODEL.PAN_SIZE, config.MODEL.PAN_SIZE)
        mspan, target = imgs[0], imgs[1]
        if config.MODEL.ORIGINAL_MS:
            #oms = mspan[0].type(torch.cuda.FloatTensor).cuda()
            oms = mspan[0].to(torch.float32).to(device)
            #opan = mspan[1].type(torch.cuda.FloatTensor).cuda().unsqueeze(1)
            opan = mspan[1].to(torch.float32).to(device).unsqueeze(1)
            inp_imgs = (oms, opan)
        elif config.TRAIN.TYPE=="RRHP":
            ms_hp = mspan[0].to(torch.float32).to(device)
            pan_hp = mspan[1].to(torch.float32).to(device).unsqueeze(1)
            ms_up = mspan[2].to(torch.float32).to(device)
            inp_imgs = (ms_hp, pan_hp, ms_up)
        else:
            inp_imgs = mspan.to(torch.float32).to(device)
            oms = inp_imgs[:, :config.MODEL.NUM_MS_BANDS, :, :]
            opan = inp_imgs[:, config.MODEL.NUM_MS_BANDS, :, :].unsqueeze(1)
        targets = target.to(torch.float32).to(device)
        outputs = model(inp_imgs)  # for aligned model, outputs include output, median_temp, aligned_temp


        if config.TRAIN.SING_LOSS:
            if config.MODEL.TYPE=="mips":
                loss_forward = criterion(outputs['pred'], targets) * 300
                latentloss = outputs["latentloss"]
                loss = loss_forward + latentloss * 0.2
            else:
                # logger.info("single loss!")
                loss = criterion(outputs, targets)
        else:
            loss = 0
            if config.TRAIN.L1:
                l1loss = criterion['l1'](outputs, targets)
                l1_meter.update(l1loss.item(), targets.size(0))
                loss += config.TRAIN.LAMBDA_L1 * l1loss
            if config.TRAIN.L2:
                l2loss = criterion['l2'](outputs, targets)
                l2_meter.update(l2loss.item(), targets.size(0))
                loss += config.TRAIN.LAMBDA_L2 * l2loss
            if config.TRAIN.SPCSIM:
                spcsim = criterion['spcsim'](outputs, targets)
                spcsim_meter.update(spcsim.item(), targets.size(0))
                loss += config.TRAIN.LAMBDA_SPCSIM * spcsim
            if config.TRAIN.BR:
                br = criterion['br'](outputs, targets)
                br_meter.update(br.item(), targets.size(0))
                loss += config.TRAIN.LAMBDA_BR * br
            if config.TRAIN.FAMSIS:
                fam_sis = criterion['fam_sis'](outputs['byp'], oms)
                famsis_meter.update(fam_sis.item(), targets.size(0))
                loss += config.TRAIN.LAMBDA_FAMSIS * fam_sis
            if config.TRAIN.PSMSIS:
                psm_sis = criterion['psm_sis'](outputs, F.interpolate(oms, scale_factor=4, mode='nearest'))
                psmsis_meter.update(psm_sis.item(), targets.size(0))
                loss += config.TRAIN.LAMBDA_PSMSIS * psm_sis
            if config.TRAIN.FAMEDGE:
                fam_edge = criterion['fam_edge'](outputs['byp'], F.interpolate(opan, scale_factor=0.25, mode='bicubic', align_corners=False, recompute_scale_factor=False))
                famedge_meter.update(fam_edge.item(), targets.size(0))
                loss += config.TRAIN.LAMBDA_FAMEDGE * fam_edge
            if config.TRAIN.PSMEDGE:
                psm_edge = criterion['psm_edge'](outputs, opan)
                psmedge_meter.update(psm_edge.item(), targets.size(0))
                loss += config.TRAIN.LAMBDA_PSMEDGE * psm_edge
            if config.TRAIN.GRADLOSS:
                gradloss = criterion['gradloss'](outputs, opan)
                gradloss_meter.update(gradloss.item(), targets.size(0))
                loss += config.TRAIN.LAMBDA_GRADLOSS * gradloss

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS

            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.CLIP_GRAD
                )
            else:
                grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.zero_grad()
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.CLIP_GRAD
                )
            else:
                grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        if device == "cuda":
            torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()
        writer.add_scalar('STEP/loss', loss.item(), global_steps)

        if idx and idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            if device == "cuda":
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            else:
                memory_used = 0.0
            etas = batch_time.avg * (num_steps - idx)
            subloss = ''
            if not config.TRAIN.SING_LOSS:
                # logger.info("multi loss!")
                if config.TRAIN.L1:
                    subloss += f"l1loss {l1_meter.val:.4f} ({l1_meter.avg:.4f})\t"
                if config.TRAIN.L2:
                    subloss += f"l2loss {l2_meter.val:.4f} ({l2_meter.avg:.4f})\t"
                if config.TRAIN.SPCSIM:
                    subloss += f"spcsimloss {spcsim_meter.val:.4f} ({spcsim_meter.avg:.4f})\t"
                if config.TRAIN.BR:
                    subloss += f"brloss {br_meter.val:.4f} ({br_meter.avg:.4f})\t"
                if config.TRAIN.FAMSIS:
                    subloss += f"famsisloss {famsis_meter.val:.4f} ({famsis_meter.avg:.4f})\t"
                if config.TRAIN.PSMSIS:
                    subloss += f"psmsisloss {psmsis_meter.val:.4f} ({psmsis_meter.avg:.4f})\t"
                if config.TRAIN.FAMEDGE:
                    subloss += f"famedgeloss {famedge_meter.val:.4f} ({famedge_meter.avg:.4f})\t"
                if config.TRAIN.PSMEDGE:
                    subloss += f"psmedgeloss {psmedge_meter.val:.4f} ({psmedge_meter.avg:.4f})\t"
                if config.TRAIN.GRADLOSS:
                    subloss += f"gradloss {gradloss_meter.val:.4f} ({gradloss_meter.avg:.4f})\t"
            # val is the value at present iteration, avg is the average over the whole epoch
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                f"batch_time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t{subloss}"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )
        # batch_size = len(outputs)
        # for i in range(batch_size):
        #    pred_img = np.transpose(outputs.data.cpu().numpy()[i], (1, 2, 0))
        #    io.imsave(f'./pre{idx}_{i}.tif', pred_img[:, :, 0:3].astype(np.uint8))

        writer_dict['train_global_steps'] = global_steps + 1

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
    )
    writer.add_scalar('EPOCH/loss', loss_meter.avg, epoch)
    if not config.TRAIN.SING_LOSS:
        if config.TRAIN.L1:
            writer.add_scalar('EPOCH/l1loss', l1_meter.avg, epoch)
        if config.TRAIN.L2:
            writer.add_scalar('EPOCH/l2loss', l2_meter.avg, epoch)
        if config.TRAIN.SPCSIM:
            writer.add_scalar('EPOCH/spcsimloss', spcsim_meter.avg, epoch)
        if config.TRAIN.BR:
            writer.add_scalar('EPOCH/brloss', br_meter.avg, epoch)
        if config.TRAIN.FAMSIS:
            writer.add_scalar('EPOCH/famsisloss', famsis_meter.avg, epoch)
        if config.TRAIN.PSMSIS:
            writer.add_scalar('EPOCH/psmsisloss', psmsis_meter.avg, epoch)
        if config.TRAIN.FAMEDGE:
            writer.add_scalar('EPOCH/famedgeloss', famedge_meter.avg, epoch)
        if config.TRAIN.PSMEDGE:
            writer.add_scalar('EPOCH/psmedgeloss', psmedge_meter.avg, epoch)
        if config.TRAIN.GRADLOSS:
            writer.add_scalar('EPOCH/gradloss', gradloss_meter.avg, epoch)


def train_ft_lwfs(
        config,
        model,
        model_o,
        criterion,
        data_loader,
        optimizer,
        epoch,
        lr_scheduler,
        logger,
        writer_dict,
        device
):
    writer = writer_dict['writer']
    model.train()
    model_o.eval()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    sis_meter = AverageMeter()
    grad_meter = AverageMeter()
    if config.TRAIN.LWF:
        lwf_real_meter = AverageMeter()
        lwf_syn_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (syn_train_data, real_train_data) in enumerate(data_loader):
        global_steps = writer_dict['train_global_steps']
        syn_mspan, syn_gt = syn_train_data[0], syn_train_data[1]
        real_mspan = real_train_data[0]
        if config.MODEL.ORIGINAL_MS:
            syn_ms = syn_mspan[0].to(torch.float32).to(device)
            syn_pan = syn_mspan[1].to(torch.float32).to(device).unsqueeze(1)
            syn_inp = (syn_ms, syn_pan)
            real_ms = real_mspan[0].to(torch.float32).to(device)
            real_pan = real_mspan[1].to(torch.float32).to(device).unsqueeze(1)
            real_inp = (real_ms, real_pan)
        else:
            syn_inp = syn_mspan.to(torch.float32).to(device)
            syn_ms = syn_inp[:, :config.MODEL.NUM_MS_BANDS, :, :]
            syn_pan = syn_inp[:, config.MODEL.NUM_MS_BANDS, :, :].unsqueeze(1)
            real_inp = real_mspan.to(torch.float32).to(device)
            real_ms = real_train_data[2].to(device)
            real_pan = real_inp[:, config.MODEL.NUM_MS_BANDS, :, :].unsqueeze(1)
        real_ms_up = F.interpolate(real_ms, scale_factor=4, mode='nearest')
        syn_gt = syn_gt.to(torch.float32).to(device)
        out_syn, aligned_syn = model(syn_inp)  # for aligned model, outputs include output, median_temp, aligned_temp
        out_syn_o, aligned_syn_o = model_o(syn_inp)

        out_real, aligned_real = model(real_inp)
        out_real_o, aligned_real_o = model_o(real_inp)

        sis_loss =  criterion['sis'](out_real, real_ms_up)
        sis_meter.update(sis_loss.item(), out_real.size(0))
        grad_loss = criterion['grad'](out_real, real_pan)
        grad_meter.update(grad_loss.item(), out_real.size(0))
        loss = config.TRAIN.LAMBDA_SIS * sis_loss + config.TRAIN.LAMBDA_GRAD * grad_loss
        if config.TRAIN.LWF:
            lwf_real = criterion['lwf_real'](out_real, out_real_o, real_pan) if config.TRAIN.MMLWF else criterion['lwf_real'](out_real, out_real_o)
            lwf_real_meter.update(lwf_real.item(), out_real.size(0))
            lwf_syn = criterion['lwf_syn'](out_syn, out_syn_o, syn_pan) if config.TRAIN.MMLWF else criterion['lwf_syn'](out_syn, out_syn_o)
            lwf_syn_meter.update(lwf_syn.item(), out_syn.size(0))
            loss = loss + config.TRAIN.LAMBDA_LWFR * lwf_real + config.TRAIN.LAMBDA_LWFS * lwf_syn

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS

            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.CLIP_GRAD
                )
            else:
                grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.zero_grad()
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.CLIP_GRAD
                )
            else:
                grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        if device == "cuda":
            torch.cuda.synchronize()

        loss_meter.update(loss.item(), out_real.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()
        writer.add_scalar('FTSTEP/loss', loss.item(), global_steps)

        if idx and idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            if device == "cuda":
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            else:
                memory_used = 0.0
            etas = batch_time.avg * (num_steps - idx)
            lwf_loss = ''
            if config.TRAIN.LWF:
                lwf_loss = f"lwf_real {lwf_real_meter.val:.4f} ({lwf_real_meter.avg:.4f})\t"
                lwf_loss += f"lwf_syn {lwf_syn_meter.val:.4f} ({lwf_syn_meter.avg:.4f})\t"
            # val is the value at present iteration, avg is the average over the whole epoch
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                f"batch_time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"sisloss {sis_meter.val:.4f} ({sis_meter.avg:.4f})\t"
                f"gradloss {grad_meter.val:.4f} ({grad_meter.avg:.4f})\t{lwf_loss}"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )
        writer_dict['train_global_steps'] = global_steps + 1

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
    )
    writer.add_scalar('FTEPOCH/loss', loss_meter.avg, epoch)
    writer.add_scalar('FTEPOCH/sisloss', sis_meter.avg, epoch)
    writer.add_scalar('FTEPOCH/gradloss', grad_meter.avg, epoch)
    if config.TRAIN.LWF:
        writer.add_scalar('FTEPOCH/lwf_real', lwf_real_meter.avg, epoch)
        writer.add_scalar('FTEPOCH/lwf_syn', lwf_syn_meter.avg, epoch)


def train_one_epoch_ft(
        config,
        model,
        model_o,
        criterion,
        data_loader,
        optimizer,
        epoch,
        lr_scheduler,
        logger,
        writer_dict,
        device
):
    writer = writer_dict['writer']
    model.train()
    model_o.eval()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    # qnr_meter = AverageMeter()
    sis_meter = AverageMeter()
    grad_meter = AverageMeter()
    if config.TRAIN.LWF:
        lwf_real_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, real_train_data in enumerate(data_loader):
        global_steps = writer_dict['train_global_steps']
        real_mspan = real_train_data[0]
        if config.MODEL.ORIGINAL_MS:
            real_ms = real_mspan[0].to(torch.float32).to(device)
            real_pan = real_mspan[1].to(torch.float32).to(device).unsqueeze(1)
            real_inp = (real_ms, real_pan)
        else:
            real_inp = real_mspan.to(torch.float32).to(device)
            real_ms = real_train_data[2].to(device)
            real_pan = real_inp[:, config.MODEL.NUM_MS_BANDS, :, :].unsqueeze(1)
        real_ms_up = F.interpolate(real_ms, scale_factor=4, mode='nearest')

        out_real, aligned_real = model(real_inp)
        out_real_o, aligned_real_o = model_o(real_inp)

        # qnr_loss = criterion['qnr'](out_real, real_pan, real_ms)
        # qnr_meter.update(qnr_loss.item(), out_real.size(0))
        sis_loss =  criterion['sis'](out_real, real_ms_up)
        sis_meter.update(sis_loss.item(), out_real.size(0))
        grad_loss = criterion['grad'](out_real, real_pan)
        grad_meter.update(grad_loss.item(), out_real.size(0))
        loss = config.TRAIN.LAMBDA_SIS * sis_loss + config.TRAIN.LAMBDA_GRAD * grad_loss
        if config.TRAIN.LWF:
            lwf_real = criterion['lwf_real'](out_real, out_real_o, real_pan) if config.TRAIN.MMLWF else criterion['lwf_real'](out_real, out_real_o)
            lwf_real_meter.update(lwf_real.item(), out_real.size(0))
            loss += config.TRAIN.LAMBDA_LWFR * lwf_real
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS

            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.CLIP_GRAD
                )
            else:
                grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.zero_grad()
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.CLIP_GRAD
                )
            else:
                grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        if device == "cuda":
            torch.cuda.synchronize()

        loss_meter.update(loss.item(), out_real.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()
        writer.add_scalar('FTSTEP/loss', loss.item(), global_steps)

        if idx and idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            if device == "cuda":
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            else:
                memory_used = 0.0
            etas = batch_time.avg * (num_steps - idx)
            lwf_loss = ''
            if config.TRAIN.LWF:
                lwf_loss = f"lwf_real {lwf_real_meter.val:.4f} ({lwf_real_meter.avg:.4f})\t"
            # val is the value at present iteration, avg is the average over the whole epoch
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                f"batch_time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"sisloss {sis_meter.val:.4f} ({sis_meter.avg:.4f})\t"
                f"gradloss {grad_meter.val:.4f} ({grad_meter.avg:.4f})\t{lwf_loss}"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )
        writer_dict['train_global_steps'] = global_steps + 1

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
    )
    writer.add_scalar('FTEPOCH/loss', loss_meter.avg, epoch)
    writer.add_scalar('FTEPOCH/sisloss', sis_meter.avg, epoch)
    writer.add_scalar('FTEPOCH/gradloss', grad_meter.avg, epoch)
    if config.TRAIN.LWF:
        writer.add_scalar('FTEPOCH/lwf_real', lwf_real_meter.avg, epoch)


@torch.no_grad()
def validate(config, data_loader, model, criterion, logger, epoch=-1, writer_dict=None, device="cpu"):
    if writer_dict is not None:
        writer = writer_dict['writer']
    model.eval()
    loss_meter = AverageMeter()
    ergas_meter = AverageMeter()
    sam_meter = AverageMeter()

    for idx, imgs in enumerate(data_loader):
        mspan, target = imgs[0], imgs[1]
        if config.MODEL.ORIGINAL_MS:
            oms = mspan[0].to(torch.float32).to(device)
            opan = mspan[1].to(torch.float32).to(device).unsqueeze(1)
            inp_imgs = (oms, opan)
        elif config.TRAIN.TYPE=="RRHP":
            ms_hp = mspan[0].to(torch.float32).to(device)
            pan_hp = mspan[1].to(torch.float32).to(device).unsqueeze(1)
            ms_up = mspan[2].to(torch.float32).to(device)
            inp_imgs = (ms_hp, pan_hp, ms_up)
        else:
            inp_imgs = mspan.to(torch.float32).to(device)
            oms = inp_imgs[:, :config.MODEL.NUM_MS_BANDS, :, :]
            opan = inp_imgs[:, config.MODEL.NUM_MS_BANDS, :, :].unsqueeze(1)
        target = target.to(torch.float32).to(device)
        output = model(inp_imgs)

        # measure accuracy and record loss
        if config.TRAIN.SING_LOSS:
            if config.MODEL.TYPE=="mips":
                loss_forward = criterion(output['pred'], target) * 300
                latentloss = output["latentloss"]
                loss = loss_forward + latentloss * 0.2
                output = output['pred']
            else:
                loss = criterion(output, target)
        else:
            loss = 0
            if config.TRAIN.L1:
                l1loss = criterion['l1'](output, target)
                loss += config.TRAIN.LAMBDA_L1 * l1loss
            if config.TRAIN.L2:
                l2loss = criterion['l2'](output, target)
                loss += config.TRAIN.LAMBDA_L2 * l2loss
            if config.TRAIN.SPCSIM:
                spcsim = criterion['spcsim'](output, target)
                loss += config.TRAIN.LAMBDA_SPCSIM * spcsim
            if config.TRAIN.BR:
                br = criterion['br'](output, target)
                loss += config.TRAIN.LAMBDA_BR * br
            if config.TRAIN.FAMSIS:
                fam_sis = criterion['fam_sis'](output['byp'], oms)
                loss += config.TRAIN.LAMBDA_FAMSIS * fam_sis
            if config.TRAIN.PSMSIS:
                psm_sis = criterion['psm_sis'](output, F.interpolate(oms, scale_factor=4, mode='nearest'))
                loss += config.TRAIN.LAMBDA_PSMSIS * psm_sis
            if config.TRAIN.FAMEDGE:
                fam_edge = criterion['fam_edge'](output['byp'], F.interpolate(opan, scale_factor=0.25, mode='bicubic', align_corners=False, recompute_scale_factor=False))
                loss += config.TRAIN.LAMBDA_FAMEDGE * fam_edge
            if config.TRAIN.PSMEDGE:
                psm_edge = criterion['psm_edge'](output, opan)
                loss += config.TRAIN.LAMBDA_PSMEDGE * psm_edge
        if config.DATA.DATASET == "wv3":
            ergas_val = avg_metric(target, output, ergas, 'wv3')
            sam_val = avg_metric(target, output, sam, 'wv3')
        else:
            ergas_val = avg_metric(target, output, ergas)
            sam_val = avg_metric(target, output, sam)

        loss_meter.update(loss.item(), target.size(0))
        ergas_meter.update(ergas_val.item(), target.size(0))
        sam_meter.update(sam_val.item(), target.size(0))

    logger.info(
        f" * Loss {loss_meter.avg:.3f} ERGAS {ergas_meter.avg:.4f} SAM {sam_meter.avg:.4f} || @ epoch {epoch}."
    )
    if writer_dict is not None:
        writer.add_scalar('METRICS/LOSS', loss_meter.avg, epoch)
        writer.add_scalar('METRICS/ERGAS', ergas_meter.avg, epoch)
        writer.add_scalar('METRICS/SAM', sam_meter.avg, epoch)
    return loss_meter.avg, ergas_meter.avg, sam_meter.avg


@torch.no_grad()
def validatefr(config, data_loader, model, criterion, logger, epoch=-1, writer_dict=None, device="cpu"):
    if writer_dict is not None:
        writer = writer_dict['writer']
    model.eval()

    loss_meter = AverageMeter()
    qnr_meter = AverageMeter()
    dl_meter = AverageMeter()
    ds_meter = AverageMeter()

    for idx, imgs in enumerate(data_loader):
        mspan, target = imgs[0], imgs[1]
        if config.MODEL.ORIGINAL_MS:
            oms = mspan[0].to(torch.float32).to(device)
            opan = mspan[1].to(torch.float32).to(device).unsqueeze(1)
            inp_imgs = (oms, opan)
        else:
            inp_imgs = mspan.to(torch.float32).to(device)
            oms = inp_imgs[:, :config.MODEL.NUM_MS_BANDS, :, :]
            opan = inp_imgs[:, config.MODEL.NUM_MS_BANDS, :, :].unsqueeze(1)
        target = target.to(torch.float32).to(device)
        output, aligned = model(inp_imgs)

        if config.TRAIN.SING_LOSS:
            loss = criterion(output, target)
        else:
            loss = 0
            if config.TRAIN.L1:
                l1loss = config.TRAIN.LAMBDA_L1 * criterion['l1'](output, target)
                loss += l1loss
            if config.TRAIN.L2:
                l2loss = config.TRAIN.LAMBDA_L2 * criterion['l2'](output, target)
                loss += l2loss
            if config.TRAIN.SPCSIM:
                spcsim = config.TRAIN.LAMBDA_SPCSIM * criterion['spcsim'](output, target)
                loss += spcsim
            if config.TRAIN.BR:
                br = config.TRAIN.LAMBDA_BR * criterion['br'](output, target)
                loss += br
            if config.TRAIN.FAMSIS:
                fam_sis = config.TRAIN.LAMBDA_FAMSIS * criterion['fam_sis'](aligned, oms)
                loss += fam_sis
            if config.TRAIN.PSMSIS:
                psm_sis = config.TRAIN.LAMBDA_PSMSIS * criterion['psm_sis'](output,
                                                                            F.interpolate(oms, scale_factor=4,
                                                                                          mode='nearest'))
                loss += psm_sis
            if config.TRAIN.FAMEDGE:
                fam_edge = config.TRAIN.LAMBDA_FAMEDGE * criterion['fam_edge'](aligned,
                                                                               F.interpolate(opan, scale_factor=0.25,
                                                                                             mode='bicubic',
                                                                                             align_corners=False,
                                                                                             recompute_scale_factor=False))
                loss += fam_edge
            if config.TRAIN.PSMEDGE:
                psm_edge = config.TRAIN.LAMBDA_PSMEDGE * criterion['psm_edge'](output, opan)
                loss += psm_edge
            if config.TRAIN.GRADLOSS:
                gradloss = config.TRAIN.LAMBDA_GRADLOSS * criterion['gradloss'](output, opan)
                loss += gradloss
        qnr_val = qnr(output, opan, oms, getAll=True)

        loss_meter.update(loss.item(), target.size(0))
        qnr_meter.update(qnr_val[0].item(), target.size(0))
        dl_meter.update(qnr_val[1].item(), target.size(0))
        ds_meter.update(qnr_val[2].item(), target.size(0))

    logger.info(
        f" * Loss {loss_meter.avg:.3f} QNR {qnr_meter.avg:.4f} D_lambda {dl_meter.avg:.4f} D_S {ds_meter.avg:.4f}  || @ epoch {epoch}."
    )
    if writer_dict is not None:
        writer.add_scalar('METRICS/LOSS', loss_meter.avg, epoch)
        writer.add_scalar('METRICS/QNR', qnr_meter.avg, epoch)
    return loss_meter.avg, qnr_meter.avg


@torch.no_grad()
def validateft(config, data_loader, model, model_o, criterion, logger, epoch=-1, writer_dict=None, device="cpu"):
    if writer_dict is not None:
        writer = writer_dict['writer']
    model.eval()
    model_o.eval()

    loss_meter = AverageMeter()
    qnr_meter = AverageMeter()
    dl_meter = AverageMeter()
    ds_meter = AverageMeter()

    for idx, real_train_data in enumerate(data_loader):
        real_mspan = real_train_data[0]
        if config.MODEL.ORIGINAL_MS:
            real_ms = real_mspan[0].to(torch.float32).to(device)
            real_pan = real_mspan[1].to(torch.float32).to(device).unsqueeze(1)
            real_inp = (real_ms, real_pan)
        else:
            real_inp = real_mspan.to(torch.float32).to(device)
            real_ms = real_train_data[2].to(device)
            real_pan = real_inp[:, config.MODEL.NUM_MS_BANDS, :, :].unsqueeze(1)
        real_ms_up = F.interpolate(real_ms, scale_factor=4, mode='nearest')

        out_real, aligned_real = model(real_inp)
        out_real_o, aligned_real_o = model_o(real_inp)

        sis_loss = criterion['sis'](out_real, real_ms_up)
        grad_loss = criterion['grad'](out_real, real_pan)
        loss = config.TRAIN.LAMBDA_SIS * sis_loss + config.TRAIN.LAMBDA_GRAD * grad_loss
        if config.TRAIN.LWF:
            lwf_real = criterion['lwf_real'](out_real, out_real_o, real_pan) if config.TRAIN.MMLWF else criterion['lwf_real'](out_real, out_real_o)
            loss += config.TRAIN.LAMBDA_LWFR * lwf_real

        qnr_val = qnr(out_real, real_pan, real_ms, getAll=True)

        loss_meter.update(loss.item(), out_real.size(0))
        qnr_meter.update(qnr_val[0].item(), out_real.size(0))
        dl_meter.update(qnr_val[1].item(), out_real.size(0))
        ds_meter.update(qnr_val[2].item(), out_real.size(0))

    logger.info(
        f" * Loss {loss_meter.avg:.3f} QNR {qnr_meter.avg:.4f} D_lambda {dl_meter.avg:.4f} D_S {ds_meter.avg:.4f}  || @ epoch {epoch}."
    )
    if writer_dict is not None:
        writer.add_scalar('FTMETRICS/LOSS', loss_meter.avg, epoch)
        writer.add_scalar('FTMETRICS/QNR', qnr_meter.avg, epoch)
    return loss_meter.avg, qnr_meter.avg


@torch.no_grad()
def throughput(config, data_loader, model, logger):
    model.eval()

    for idx, imgs in enumerate(data_loader):
        mspan = imgs[0]
        batch_size = config.DATA.BATCH_SIZE
        if config.MODEL.ORIGINAL_MS:
            oms = mspan[0].to(torch.float32).cuda()
            opan = mspan[1].to(torch.float32).cuda().unsqueeze(1)
            inp_imgs = (oms, opan)
        elif config.TRAIN.TYPE=="RRHP":
            ms_hp = mspan[0].to(torch.float32).cuda()
            pan_hp = mspan[1].to(torch.float32).cuda().unsqueeze(1)
            ms_up = mspan[2].to(torch.float32).cuda()
            inp_imgs = (ms_hp, pan_hp, ms_up)
        else:
            inp_imgs = mspan.to(torch.float32).cuda()
        for i in range(50):
            model(inp_imgs)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 100 times")
        tic1 = time.time()
        for i in range(100):
            model(inp_imgs)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(
            f"batch_size {batch_size} throughput {100 * batch_size / (tic2 - tic1)} time_per_img {(tic2 - tic1) / (100 * batch_size)}"
        )
        return

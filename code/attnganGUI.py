import os
import pickle
from tkinter import scrolledtext
import numpy as np
import torch
from torch.autograd import Variable
import tkinter as tk
from PIL import Image, ImageTk
from nltk import RegexpTokenizer
from miscc.config import cfg, cfg_from_file
from miscc.utils import build_super_images, build_super_images2, mkdir_p
from model import RNN_ENCODER, G_NET


def load_text_data(data_dir):
    filepath = os.path.join(data_dir, 'captions.pickle')

    with open(filepath, 'rb') as f:
        x = pickle.load(f)
        train_captions, test_captions = x[0], x[1]
        ixtoword, wordtoix = x[2], x[3]
        del x
        n_words = len(ixtoword)

    captions = test_captions
    return captions, ixtoword, wordtoix, n_words


def get_data_dict(text, wordtoix):
    data_dic = {}
    sentences = text.split('\n')
    # a list of indices for a sentence
    captions = []
    cap_lens = []
    for sent in sentences:
        if len(sent) == 0:
            continue
        sent = sent.replace("\ufffd\ufffd", " ")
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sent.lower())
        if len(tokens) == 0:
            print('sent', sent)
            continue

        rev = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0 and t in wordtoix:
                rev.append(wordtoix[t])
        captions.append(rev)
        cap_lens.append(len(rev))

    max_len = np.max(cap_lens)

    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype='int64')
    for i in range(len(captions)):
        idx = sorted_indices[i]
        cap = captions[idx]
        c_len = len(cap)
        cap_array[i, :c_len] = cap
    data_dic[0] = [cap_array, cap_lens, sorted_indices]

    return data_dic


def gen_img(data_dic, n_words, ixtoword, save_path='../default_output'):
    show_img = None

    text_encoder = RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder = text_encoder.cuda()
    text_encoder.eval()

    netG = G_NET()
    state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
    netG.load_state_dict(state_dict)
    netG.cuda()
    netG.eval()

    for key in data_dic:
        save_dir = '%s/%s' % (save_path, key)
        mkdir_p(save_dir)
        captions, cap_lens, sorted_indices = data_dic[key]

        batch_size = captions.shape[0]
        nz = cfg.GAN.Z_DIM

        with torch.no_grad():
            captions = Variable(torch.from_numpy(captions))
            cap_lens = Variable(torch.from_numpy(cap_lens))

            captions = captions.cuda()
            cap_lens = cap_lens.cuda()

        for i in range(1):  # 16
            with torch.no_grad():
                noise = Variable(torch.FloatTensor(batch_size, nz))
                noise = noise.cuda()

            hidden = text_encoder.init_hidden(batch_size)
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            mask = (captions == 0)

            noise.data.normal_(0, 1)
            fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
            # G attention
            cap_lens_np = cap_lens.cpu().data.numpy()
            for j in range(batch_size):
                save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                for k in range(len(fake_imgs)):
                    im = fake_imgs[k][j].data.cpu().numpy()
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))
                    im = Image.fromarray(im)
                    fullpath = '%s_g%d.png' % (save_name, k)
                    im.save(fullpath)

                    # GUI只显示第一个句子的256*256图片
                    if j == 0 and k == len(fake_imgs) - 1:
                        im.save('../finaloutput/final.png')
                        show_img = im

                # attention maps
                for k in range(len(attention_maps)):
                    if len(fake_imgs) > 1:
                        im = fake_imgs[k + 1].detach().cpu()
                    else:
                        im = fake_imgs[0].detach().cpu()
                    attn_maps = attention_maps[k]
                    att_sze = attn_maps.size(2)
                    img_set, sentences = build_super_images2(im[j].unsqueeze(0), captions[j].unsqueeze(0),
                                                             [cap_lens_np[j]], ixtoword, [attn_maps[j]], att_sze)
                    if img_set is not None:
                        im = Image.fromarray(img_set)
                        fullpath = '%s_a%d.png' % (save_name, k)
                        im.save(fullpath)

    return show_img


def generate():
    # 确定生成图片类型，加载相应的模型
    selected_option = selected_value.get()
    if selected_option == 1:  # 鸟
        data_dir = '../data/birds'
        cfg_path = 'cfg/eval_bird.yml'
    elif selected_option == 2:  # coco
        data_dir = '../data/coco'
        cfg_path = 'cfg/eval_coco.yml'

    # 读取文本描述
    text_content = text_description.get("1.0", tk.END)

    # 读取字典资料
    _, ixtoword, wordtoix, n_words = load_text_data(data_dir)

    # 生成图片
    cfg_from_file(cfg_path)
    data_dict = get_data_dict(text_content, wordtoix)
    ret_img = gen_img(data_dict, n_words, ixtoword)
    ret_img = ret_img.resize((256, 256))
    show_img = ImageTk.PhotoImage(ret_img)


    img = Image.open("../finaloutput/final.png")
    img = img.resize((256, 256))
    fake_img = ImageTk.PhotoImage(img)
    h, w = 256, 256
    canvas = tk.Canvas(root, bg='white', height=h, width=w)
    canvas.grid(row=5, column=0, padx=10, pady=5)
    # 存储 fake_img 到 canvas 的属性，以保持引用
    canvas.image = fake_img
    image = canvas.create_image(w / 2, h / 2, anchor=tk.CENTER, image=fake_img)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Text2Img-based AttnGAN")

    """选择生成的图片类型，以选择载入的模型"""
    label_select = tk.Label(root, text='Choose the picture type', fg='red')
    label_select.grid(row=0, column=0, padx=5, pady=10, sticky='w')

    selected_value = tk.IntVar()
    radio1 = tk.Radiobutton(root, text="birds", value=1, variable=selected_value)
    radio1.grid(row=1, column=0, padx=10, pady=5, sticky='w')
    radio2 = tk.Radiobutton(root, text="common objects", value=2, variable=selected_value)
    radio2.grid(row=1, column=1, padx=10, pady=5, sticky='w')

    """描述文本框"""
    label_description = tk.Label(root, text='Description', fg='red')
    label_description.grid(row=2, column=0, padx=10, pady=5, sticky='w')

    text_description = tk.scrolledtext.ScrolledText(root, width=70, height=10)
    text_description.grid(row=3, column=0, columnspan=3, padx=10, pady=5)

    """生成图片"""
    button_gen = tk.Button(root, text='generate', command=lambda: generate())
    button_gen.grid(row=4, column=2, padx=10, pady=5)

    """图片预览"""
    label_img = tk.Label(root, text='Generated Image ', fg='red')
    label_img.grid(row=4, column=0, padx=10, pady=5, sticky='w')

    root.mainloop()

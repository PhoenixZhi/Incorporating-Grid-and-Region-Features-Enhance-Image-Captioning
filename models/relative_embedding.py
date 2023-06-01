import torch


def BoxRelationalEmbedding(f_g, mask=None, dim_g=64, wave_len=1000, trignometric_embedding=True):
    batch_size = f_g.size(0)
    reg_cx, reg_cy, reg_w, reg_h = torch.chunk(f_g, 4, dim=-1)
    if mask == None:
        delta_x = reg_cx - reg_cx.view(batch_size, 1, -1)
        delta_x = delta_x / reg_w
        for m in range(delta_x.shape[-1]):
            delta_x[:, m, m] = reg_cx[:, m, 0]

        delta_y = reg_cy - reg_cy.view(batch_size, 1, -1)
        delta_y = delta_y / reg_h
        for m in range(delta_y.shape[-1]):
            delta_y[:, m, m] = reg_cy[:, m, 0]

        w_ = reg_w / reg_w.view(batch_size, 1, -1)
        h_ = reg_h / reg_h.view(batch_size, 1, -1)
        for m in range(delta_y.shape[-1]):
            w_[:, m, m] = reg_w[:, m, 0]
            h_[:, m, m] = reg_h[:, m, 0]
    else:
        grid_x_min, grid_y_min, grid_x_max, grid_y_max = get_normalized_grids(batch_size, mask=mask)

        grid_cx = (grid_x_min + grid_x_max) * 0.5
        grid_cy = (grid_y_min + grid_y_max) * 0.5
        grid_w = grid_x_max - grid_x_min
        grid_h = grid_y_max - grid_y_min
        delta_x = reg_cx - grid_cx.view(batch_size, 1, -1)
        delta_y = reg_cy - grid_cy.view(batch_size, 1, -1)
        w_ = reg_w / grid_w.view(batch_size, 1, -1)
        h_ = reg_h / grid_h.view(batch_size, 1, -1)

    delta_x = torch.clamp(torch.abs(delta_x), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = torch.clamp(torch.abs(delta_y), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w_)
    delta_h = torch.log(h_)

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return (embedding)



def GridRelationalEmbedding(batch_size, grid_size=7, dim_g=64, wave_len=1000, trignometric_embedding=True):
    # make grid
    a = torch.arange(0, grid_size).float().cuda()
    c1 = a.view(-1, 1).expand(-1, grid_size).contiguous().view(-1)
    c2 = a.view(1, -1).expand(grid_size, -1).contiguous().view(-1)
    c3 = c1 + 1
    c4 = c2 + 1
    f = lambda x: x.view(1, -1, 1).expand(batch_size, -1, -1)
    x_min, y_min, x_max, y_max = f(c1), f(c2), f(c3), f(c4)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)  # bs * r * r *4

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return (embedding)



def get_normalized_grids(bs, grid_size=[6, 10], mask=None):
    mask = mask.squeeze(1).squeeze(1).view(-1, 6, 10)
    row = 10 - torch.sum(mask[:, 0, :], dim=1)
    col = 6 - torch.sum(mask[:, :, 0], dim=1)
    b = torch.arange(0, grid_size[0]).float().cuda()
    a = torch.arange(0, grid_size[1]).float().cuda()
    c1 = a.view(1, -1).expand(grid_size[0], -1).contiguous().view(-1)
    c2 = b.view(-1, 1).expand(-1, grid_size[1]).contiguous().view(-1)
    c3 = c1 + 1
    c4 = c2 + 1
    f1 = lambda x: x.view(1, -1, 1).expand(bs, -1, -1) / col.view(-1, 1, 1).repeat(1, 60, 1)
    f2 = lambda x: x.view(1, -1, 1).expand(bs, -1, -1) / row.view(-1, 1, 1).repeat(1, 60, 1)
    x_min, y_min, x_max, y_max = f2(c1), f1(c2), f2(c3), f1(c4)
    return x_min, y_min, x_max, y_max


def AllRelationalEmbedding(f_g, dim_g=64, wave_len=1000, trignometric_embedding=True, require_all_boxes=False,
                           gri_mask=None):
    """
    Given a tensor with bbox coordinates for detected objects on each batch image,
    this function computes a matrix for each image

    with entry (i,j) given by a vector representation of the
    displacement between the coordinates of bbox_i, and bbox_j

    input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
    output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
    """
    # returns a relational embedding for each pair of bboxes, with dimension = dim_g
    # follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055

    batch_size = f_g.size(0)
    reg_cx, reg_cy, reg_w, reg_h = torch.chunk(f_g, 4, dim=-1)
    grid_x_min, grid_y_min, grid_x_max, grid_y_max = get_normalized_grids(batch_size, mask=gri_mask)

    grid_cx = (grid_x_min + grid_x_max) * 0.5
    grid_cy = (grid_y_min + grid_y_max) * 0.5
    grid_w = grid_x_max - grid_x_min
    grid_h = grid_y_max - grid_y_min
    cx = torch.cat([reg_cx, grid_cx], dim=1)
    cy = torch.cat([reg_cy, grid_cy], dim=1)
    w = torch.cat([reg_w, grid_w], dim=1)
    h = torch.cat([reg_h, grid_h], dim=1)

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)  # bs * r * r *4

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    if require_all_boxes:
        all_boxes = torch.cat([cx, cy, w, h], dim=-1)
        return (embedding), all_boxes
    return (embedding)

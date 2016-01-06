function [ overlap ] = IoU( bboxA, bboxB )
% bboxs are in the Pascal [xmin, ymin, xmax, ymax] format

I_minx = max(bboxA(1), bboxB(1));
I_miny = max(bboxA(2), bboxB(2));
I_maxx = min(bboxA(3), bboxB(3));
I_maxy = min(bboxA(4), bboxB(4));

if(I_minx < I_maxx && I_miny < I_maxy)
    I = (I_maxx - I_minx + 1) * (I_maxy - I_miny + 1);
    U = (bboxA(3) - bboxA(1) + 1) * (bboxA(4) - bboxA(2) + 1);
    U = U + ((bboxB(3) - bboxB(1) + 1) * (bboxB(4) - bboxB(2) + 1));
    U = U - I;
    overlap = I / U;
else
    overlap = 0;
end

end


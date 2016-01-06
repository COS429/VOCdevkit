% kmeans region proposals

[idx, C] = kmeans(allBoxes, 10);


hit = false(length(allBoxes),1);
for i = 1:length(hit)
    for c = 1:10
        if(IoU(C(c,:), allBoxes(i,:)) > 0.5)
            hit(i) = true;
        end
    end
end

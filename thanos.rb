
top_dicts = ["fruits-360/Test", "fruits-360/Training"]

eliminate_rate = 0.95

top_dicts.each do |top_dict|
  cates = Dir.glob("#{top_dict}/*")
  total = 0
  cates.each do |cate|
    imgs = Dir.glob("#{cate}/*.jpg")
    len = imgs.size
    step_size = len / (len * (1 - eliminate_rate)).to_i
    imgs.each_with_index do |file, idx|
      next if idx % step_size == 0
      File.delete(file)
    end
  end
end
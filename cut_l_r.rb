#!/usr/bin/env ruby
require 'rmagick'

if ARGV.size < 3
	p "usage: ruby cut_l_r.rb [input image] [output left side image] [output right side image]"
	exit 1
end

input_im_path = ARGV[0]
output_l_path = ARGV[1]
output_r_path = ARGV[2]

image = Magick::Image.read(input_im_path)[0]
height = image.rows
width = image.columns
width_2 = width/2
image.crop(0,0,width_2,height).write output_l_path
image.crop(width-width_2,0,width_2,height).write output_r_path

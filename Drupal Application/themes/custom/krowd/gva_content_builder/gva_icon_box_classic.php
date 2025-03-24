<?php 
if(!class_exists('element_gva_icon_box_classic')):
   class element_gva_icon_box_classic{
      public function render_form(){
         $option_1 = array(
            '0'   => '--Default--',
            '14'   => '14',
            '16'   => '16',
            '18'   => '18',
            '20'   => '20',
            '22'   => '22',
            '24'   => '24',
            '26'   => '26',
            '28'   => '28',
            '30'   => '30',
            '32'   => '32',
            '34'   => '34',
            '36'   => '36',
            '38'   => '38',
            '40'   => '40',
            '42'   => '42',
            '44'   => '44',
            '46'   => '46',
            '48'   => '48',
            '50'   => '50',
            '52'   => '52',
            '54'   => '54',
            '56'   => '56',
            '58'   => '58',
            '60'   => '60',
            '70'   => '70',
            '80'   => '80',
            '90'   => '90',
            '100'  => '100'
         );
         $fields = array(
            'type' => 'element_gva_icon_box_classic',
            'title' => ('Icon Box Classic'), 
            'fields' => array(
               array(
                  'id'        => 'title',
                  'type'      => 'text',
                  'title'     => t('Title'),
                  'admin'     => true
               ),
               array(
                  'id'        => 'content',
                  'type'      => 'textarea',
                  'title'     => t('Content'),
                  'desc'      => t('Some Shortcodes and HTML tags allowed'),
               ),
               array(
                  'id'            => 'hidden_content',
                  'type'          => 'select',
                  'options'       => array(
                     ''                      => t('Always Display'),
                     'hidden-xs hidden-sm'   => t('Hidden Small & Extra Small Screen (hidden-sm & hidden-xs)'), 
                     'hidden-sm'             => t('Hidden Small Screen (hidden-sm)'), 
                     'hidden-xs'             => t('hidden Extra Small Screen (hidden-xs)'), 
                  ),
                  'title'  => t('Hidden Content in Small Screen'),
               ),
               array(
                  'id'        => 'icon',
                  'type'      => 'text',
                  'title'     => t('Icon class'),
                  'std'       => '',
                  'desc'     => t('Use class icon font <a target="_blank" href="http://fontawesome.io/icons/">Icon Awesome</a> or <a target="_blank" href="http://gaviasthemes.com/icons/">Custom icon</a>'),
               ),
               array(
                  'id'        => 'image',
                  'type'      => 'upload',
                  'title'     => t('Image Icon'),
                  'desc'      => t('Use image icon instead of icon class'),
               ),
               array(
                  'id'            => 'icon_position',
                  'type'          => 'select',
                  'options'       => array(
                     'top-center'     => 'Top Center', 
                     'left v1'        => 'Left #1',
                     'left v2'        => 'Left #2',
                     'left v3'        => 'Left #3',
                  ),
                  'title'  => t('Icon Position'),
                  'std'    => 'top',
               ),
               array(
                  'id'        => 'link',
                  'type'      => 'text',
                  'title'     => t('Link'),
                  'desc'      => t('Link for text'),
                  'class'     => 'width-1-2'
               ),
                array(
                  'id'        => 'target',
                  'type'      => 'select',
                  'options'   => array( 'off' => 'No', 'on' => 'Yes' ),
                  'title'     => t('Open in new window'),
                  'class'     => 'width-1-2',
                  'desc'      => t('Adds a target="_blank" attribute to the link.'),
               ),
               array(
                  'id'        => 'box_background',
                  'type'      => 'text',
                  'title'     => t('Box Background'),
                  'desc'      => t('Box Background, e.g: #f5f5f5')
               ),
              
              array(
                 'id'        => 'info',
                 'type'      => 'info',
                 'title'      => 'Background, Color Icon'
               ),

               array(
                  'id'        => 'icon_bg_available',
                  'type'      => 'select',
                  'title'     => 'Background Icon Available',
                  'class'     => 'width-1-4',
                  'options'   => array(
                     ''             => '------',
                     'bg-theme'     => 'Background Theme',
                     'bg-black'     => 'Background Black',
                     'bg-dark'      => 'Background Dark',
                     'bg-white'     => 'Background White'
                  )
               ),
               array(
                  'id'        => 'icon_background',
                  'type'      => 'text',
                  'title'     => 'Custom Background Icon',
                  'desc'      => t('e.g: #f5f5f5'),
                  'class'     => 'width-1-4',
               ),
               array(
                  'id'        => 'icon_color_available',
                  'type'      => 'select',
                  'title'     => 'Icon Color Available',
                  'class'     => 'width-1-4',
                  'options'   => array(
                     ''                => '------',
                     'color-theme'     => 'Color Theme',
                     'color-black'     => 'Color Black',
                     'color-dark'      => 'Color Dark',
                     'color-white'     => 'Color White'
                  )
               ),
               array(
                  'id'        => 'icon_color',
                  'type'      => 'text',
                  'title'     => t('Custom Icon Color'),
                  'desc'      => t('e.g: #f5f5f5'),
                  'class'     => 'width-1-4',
               ),

               array(
                  'id'        => 'icon_width',
                  'type'      => 'select',
                  'title'     => t('Icon Width'),
                  'options'   => array(
                     'fa-1x'  => t('Fa 1x small'), 
                     'fa-2x'  => t('Fa 2x'), 
                     'fa-3x'  => t('Fa 3x'),
                     'fa-4x'  => t('Fa 4x'),
                     'fa-5x'  => t('Fa 5x'),
                     'width-full'  => t('Width 100%'),
                  ),
                  'class'     => 'width-1-4',
               ),
               array(
                  'id'        => 'icon_radius',
                  'type'      => 'select',
                  'title'     => t('Icon Radius'),
                  'options'   => array(
                     ''           => t('--None--'), 
                     'radius-1x'  => t('Radius 1x'), 
                     'radius-2x'  => t('Radius 2x'),
                     'radius-5x'  => t('Radius 5x'),
                  ),
                  'class'     => 'width-1-4',
               ),
               array(
                  'id'        => 'icon_border',
                  'type'      => 'select',
                  'title'     => t('Icon Border'),
                  'options'   => array(
                     ''           => t('--None--'), 
                     'border-1'  => t('Border 1px'), 
                     'border-2'  => t('Border 2px'),
                     'border-3'  => t('Border 3px'),
                     'border-4'  => t('Border 4px'),
                     'border-5'  => t('Border 5px'),
                     'border-s1' => t('Border Padding'),
                  ),
                  'class'     => 'width-1-4',
               ),
               array(
                  'id'        => 'border_color',
                  'type'      => 'select',
                  'title'     => 'Border Color',
                  'class'     => 'width-1-4',
                  'options'   => array(
                     ''                => '------',
                     'theme'     => 'Color Theme',
                     'black'     => 'Color Black',
                     'dark'      => 'Color Dark',
                     'white'     => 'Color White'
                  ),
                  'class'     => 'width-1-4',
               ),
               array(
                  'id'        => 'title_font_size',
                  'type'      => 'select',
                  'title'     => t('Title Font Size'),
                  'options'   => $option_1,
                  'default'   => '0',
                  'class'     => 'width-1-4',
               ),
               array(
                  'id'        => 'title_line_height',
                  'type'      => 'select',
                  'title'     => t('Title Line Height'),
                  'options'   => $option_1,
                  'default'   => '0',
                  'class'     => 'width-1-4',
               ),
               array(
                  'id'        => 'title_color',
                  'type'      => 'select',
                  'title'     => t('Title Color'),
                  'options'   => array(
                     'text-black'   => 'Black Color',
                     'text-white'   => 'White Color',
                     'text-theme'   => 'Theme Color'
                  ),
                  'default'   => 'text-black',
                  'class'     => 'width-1-4'
               ),
               array(
                  'id'        => 'desc_color',
                  'type'      => 'select',
                  'title'     => t('Description Color'),
                  'options'   => array(
                     'text-gray'    => 'Gray Color',
                     'text-gray-light'   => 'Gray Light Color',
                     'text-black'   => 'Black Color',
                     'text-white'   => 'White Color',
                     'text-theme'   => 'Theme Color'
                  ),
                  'default'   => 'text-gray',
                  'class'     => 'width-1-4',
               ),
               array(
                  'id'        => 'box_shadow',
                  'type'      => 'select',
                  'title'     => 'Icon Box Shadow',
                  'options'   => array(
                     ''             => t('Disable Box Shadow'), 
                     'icon-shadow'  => t('Enable Box Shadow')
                  ),
                  'class'     => 'width-1-4',
               ),
               array(
                  'id'        => 'vertical_align_content',
                  'type'      => 'select',
                  'title'     => 'Verticle Align content',
                  'options'   => array(
                     'top'    => t('Top'), 
                     'middle' => t('Middle'),
                     'bottom' => t('Bottom')
                  ),
                  'class'     => 'width-1-4',
               ),
               array(
                  'id'        => 'margin',
                  'type'      => 'select',
                  'title'     => t('Margin Bottom'),
                  'options'   => array(
                     'box-margin-0'       => t('Remove Margin Bottom'), 
                     'box-margin-small'   => t('Margin Bottom Small'), 
                     'box-margin-medium'  => t('Margin Bottom Medium'),
                     'box-margin-large'   => t('Margin Bottom Large'),
                  ),
                  'class'     => 'width-1-4',
                  'default'   => 'box-margin-small'
               ),
               
               array(
                  'id'        => 'effect',
                  'type'      => 'select',
                  'title'     => t('Animation'),
                  'options'   => array(
                     'effect'      => t('None'), 
                     'effect-v1'   => t('Effect V1'),
                  ),
                  'class'     => 'width-1-4',
                  'default'   => ''
               ),

               array(
                  'id'        => 'animate',
                  'type'      => 'select',
                  'title'     => t('Animation'),
                  'desc'      => t('Entrance animation for element'),
                  'options'   => gavias_content_builder_animate(),
                  'class'     => 'width-1-3'
               ), 
               array(
                  'id'        => 'animate_delay',
                  'type'      => 'select',
                  'title'     => t('Animation Delay'),
                  'options'   => gavias_content_builder_delay_wow(),
                  'desc'      => '0 = default',
                  'class'     => 'width-1-3'
               ), 
               
               array(
                  'id'     => 'el_class',
                  'type'      => 'text',
                  'title'  => t('Extra class name'),
                  'desc'      => t('Style particular content element differently - add a class name and refer to it in custom CSS.'),
                  'class'     => 'width-1-3'
               ),

            ),                                       
         );
         return $fields;
      }

      public static function render_content( $attr = array(), $content = '' ){
         global $base_url;
         extract(gavias_merge_atts(array(
            'title'                       => '',
            'content'                     => '',
            'hidden_content'              => '',
            'icon'                        => '',
            'image'                       => '',
            'icon_position'               => 'top',
            'box_background'              => '',
            'icon_color_available'        => '',
            'icon_color'                  => '',
            'icon_bg_available'           => '',
            'icon_background'             => '',
            'icon_radius'                 => '',
            'icon_border'                 => '',
            'border_color'                => '',
            'margin'                      => 'box-margin-small',
            'icon_width'                  => 'fa-2x',
            'link'                        => '',
            'box_shadow'                  => '',
            'title_font_size'             => '0',
            'title_line_height'           => '0',
            'title_color'                 => 'text-dark',
            'desc_color'                  => 'text-dark',
            'vertical_align_content'      => 'top',
            'target'                      => '',
            'animate'                     => '',
            'animate_delay'               => '',
            'min_height'                  => '',
            'el_class'                    => '',
            'effect'                      => ''
         ), $attr));
         
         if($image) $image = $base_url . $image; 

         // target
         if( $target =='on' ){
            $target = 'target="_blank"';
         } else {
            $target = false;
         }

         $class = array();
         $class[] = $icon_position;
         $class[] = $margin;
         if($image) $class[] = 'icon-image';
         if($el_class) $class[] = $el_class;
         if($effect) $class[] = $effect;

         if($box_background) $class[] = 'box-background';
         if($icon_border) $class[] = 'icon-border';
         if($icon_background) $class[] = 'icon-background';
         
         if($icon_border == 'border-s1') $class[] .= " border-s1";

         $icon_class = "{$icon_width} {$icon_radius} {$icon_border}";
         if($border_color) $icon_class .= " i-border-{$border_color}";
         if($icon_border || $icon_background || $icon_bg_available) $icon_class .= ' fa-stack';

         if($icon_bg_available) $icon_class .= ' ' . $icon_bg_available;
         if($icon_color_available) $icon_class .= ' ' . $icon_color_available;

         $icon_class_inner = "";
         if($icon_border == 'border-s1'){
            $icon_class_inner .= "{$icon_radius} i-border-{$border_color}";
         }

         if($box_shadow) $icon_class .= " {$box_shadow}";

         $style = array(); // Style box
         if($min_height) $style[] = "min-height:{$min_height};";
         if($box_background) $style[] = "background-color:{$box_background};";
         
         $style_icon = ''; // Style icon
         if($icon_background) $style_icon .= "background: {$icon_background};";
         if($icon_color) $style_icon .= "color: {$icon_color};";
         if($style_icon) $style_icon = "style=\"{$style_icon}\"";

         $classes_title_text = '';
         $classes_title = array();
         $title_line_height ? $classes_title[] = "lheight-{$title_line_height}" : false;
         $title_font_size ? $classes_title[] = "fsize-{$title_font_size}" : false;
         $classes_title[] = $title_color;
         $classes_title_text = implode(' ', $classes_title);

         $classes_desc_text = '';
         $classes_desc = array();
         $classes_desc[] = $desc_color;
         $classes_desc[] = $hidden_content;
         $classes_desc_text = implode(' ', $classes_desc);

         if($animate) $class[] = 'wow ' . $animate;

         $title_html = $title;
         if($link){
            $title_html = "<a href=\"{$link}\" {$target}>{$title}</a>";
         }

         $_classes = count($class) > 0 ? implode(' ', $class) : '';
         $_style = count($style) > 0 ? 'style="' . implode(';', $style) . '"' : '';
         ob_start();
         ?>
         <div class="widget gsc-icon-box <?php print $_classes ?>" <?php print $_style ?> <?php print gavias_content_builder_print_animate_wow('', $animate_delay) ?>>
            
            <?php if($icon || $image){ ?>
               <div class="highlight-icon verticle-align-<?php print $vertical_align_content ?>">
                  <span class="icon-inner <?php echo $icon_class_inner ?>">   
                     <span class="icon-container <?php print $icon_class ?>" <?php print $style_icon ?>>
                        <?php if($icon){ ?><span class="icon <?php print $icon ?>"></span> <?php } ?>
                        <?php if($image){ ?><span class="icon"><img src="<?php print $image ?>" alt="<?php print strip_tags($title) ?>"/> </span> <?php } ?>
                     </span>
                  </span>
               </div>
            <?php } ?>

            <div class="highlight_content verticle-align-<?php print $vertical_align_content ?>">
               <?php if($title){ ?>
                  <h3 class="title <?php print $classes_title_text ?>"><?php print $title_html; ?></h3>
               <?php } ?>   
               <?php if($content){ ?>
                  <div class="desc <?php print $classes_desc_text ?>"><?php print $content; ?></div>
               <?php } ?>   
            </div>

         </div>  

         <?php return ob_get_clean() ?>
       <?php
      }

   } 
endif;   

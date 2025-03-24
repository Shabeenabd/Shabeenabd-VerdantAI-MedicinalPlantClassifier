<?php 
if(!class_exists('element_gva_counter')):
   class element_gva_counter{
      public function render_form(){
         $fields = array(
            'type' => 'gsc_counter',
            'title' => ('Counter'),
            'fields' => array(
               array(
                  'id'        => 'title',
                  'title'     => t('Title'),
                  'type'      => 'text',
                  'admin'     => true
               ),
               array(
                  'id'        => 'icon',
                  'title'     => t('Icon'),
                  'type'      => 'text',
                  'std'       => '',
                  'desc'     => t('Use class icon font <a target="_blank" href="http://fontawesome.io/icons/">Icon Awesome</a> or <a target="_blank" href="http://gaviasthemes.com/icons/">Custom icon</a>'),
               ),
               array(
                  'id'        => 'image',
                  'type'      => 'upload',
                  'title'     => t('Icon Top'),
               ),
               array(
                  'id'        => 'number',
                  'title'     => t('Number'),
                  'type'      => 'text',
               ),
               array(
                  'id'        => 'symbol',
                  'title'     => t('Symbol'),
                  'type'      => 'text',
                  'desc'      => 'e.g %'
               ),
               array(
                  'id'        => 'type',
                  'title'     => t('Style'),
                  'type'      => 'select',
                  'options'   => array(
                     'icon-top v1'      => 'Icon top #1',
                     'icon-top v2'      => 'Icon top #2',
                     'icon-top v3'      => 'Icon top #3',
                  ),
                  'std'    => 'icon-top v1',
               ),
               array(
                  'id'        => 'color',
                  'type'      => 'text',
                  'title'     => t('Icon Color'),
                  'desc'      => t('Use color name ( blue ) or hex ( #2991D6 )'),
               ),
               array(
                  'id'        => 'text_color',
                  'type'      => 'select',
                  'title'     => t('Skin Text for box'),
                  'options'   => array(
                     'text-dark'   => 'Text dark',
                     'text-light'   => 'Text light'
                  ),
                  'std'       => 'text-dark'
               ),
               array(
                  'id'        => 'el_class',
                  'type'      => 'text',
                  'title'     => t('Extra class name'),
                  'desc'      => t('Style particular content element differently - add a class name and refer to it in custom CSS.'),
               ),
               array(
                  'id'     => 'animate',
                  'type'      => 'select',
                  'title'  => t('Animation'),
                  'sub_desc'  => t('Entrance animation'),
                  'options'   => gavias_content_builder_animate(),
                  'class'     => 'width-1-2'
               ),
               array(
                  'id'        => 'animate_delay',
                  'type'      => 'select',
                  'title'     => t('Animation Delay'),
                  'options'   => gavias_content_builder_delay_aos(),
                  'desc'      => '0 = default',
                  'class'     => 'width-1-2'
               ), 
            ),                                      
         );
         return $fields;
      }

       public static function render_content( $attr = array(), $content = '' ){
         global $base_url;
         extract(gavias_merge_atts(array(
            'title'           => '',
            'icon'            => '',
            'image'           => '',
            'number'          => '',
            'symbol'          => '',
            'type'            => 'icon-top',
            'el_class'        => '',
            'text_color'      => 'text-black',
            'color'           => '',
            'animate'         => '',
            'animate_delay'   => ''
         ), $attr));
         if($image){ 
            $image = $base_url . $image;
         }
         $class = array();
         $class[] = $el_class;
         $class[] = 'position-'.$type;
         $class[] = $text_color;
       
         $style = '';
         if($color) $style = "color: {$color};";
         if($style) $style = 'style="'.$style.'"';
         if($animate) $class[] = ' wow ' . $animate;
         $_classes = count($class) > 0 ? implode(' ', $class) : '';
         ob_start();
         ?>
            <div class="widget milestone-block <?php print $_classes ?>" <?php print gavias_content_builder_print_animate_wow('', $animate_delay) ?>>
               <?php if($icon){ ?>
                  <div class="milestone-icon">
                     <div class="icon"><span <?php print $style ?> class="<?php print $icon; ?>"></span></div></div>
               <?php } ?> 
               <?php if($image){ ?>
                  <div class="milestone-image">
                     <img src="<?php print $image ?>" alt="" />
                  </div>
               <?php } ?>  
               <div class="milestone-right">
                  <div class="milestone-number-inner" <?php print $style ?>>
                     <span class="milestone-number">
                        <?php print $number; ?></span><span class="symbol"><?php print $symbol ?>
                     </span>
                  </div>
                  <div class="milestone-text"><?php print $title ?></div>
               </div>
            </div>
         <?php return ob_get_clean() ?>
         <?php
      }
   }
endif;
   




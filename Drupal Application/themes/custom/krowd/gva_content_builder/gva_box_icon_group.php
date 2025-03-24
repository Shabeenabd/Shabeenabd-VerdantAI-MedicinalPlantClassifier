<?php 

if(!class_exists('element_gva_box_icon_group')):
   class element_gva_box_icon{
      public function render_form(){
         $fields = array(
            'type' => 'gsc_box_icon_group',
            'title' => ('Box Icon Group'), 
            'fields' => array(
               array(
                  'id'        => 'title',
                  'type'      => 'text',
                  'title'     => t('Title'),
                  'admin'     => true
               ),
               array(
                  'id'        => 'sub_title',
                  'type'      => 'text',
                  'title'     => t('Sub Title'),
               ),
               array(
                  'id'        => 'content',
                  'type'      => 'textarea',
                  'title'     => t('Content'),
                  'desc'      => t('Some Shortcodes and HTML tags allowed'),
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
                  'id'        => 'layout_style',
                  'type'      => 'select',
                  'title'     => t('Layout Style'),
                  'options'   => array(
                     ''           => t('--None--'), 
                     'style-v1'  => t('Style V1'), 
                     'style-v2'  => t('Style V2'),
                     'style-v3'  => t('Style V3'),
                     'style-v4'  => t('Style V4')
                  )
               ),
               array(
                  'id'        => 'skin_text',
                  'type'      => 'select',
                  'title'     => 'Skin Text for box',
                  'options'   => array(
                     'text-dark'  => t('Text Dark'), 
                     'text-light' => t('Text Light')
                  ) 
               ),    
               array(
                  'id'        => 'box_background',
                  'type'      => 'select',
                  'title'     => 'Background',
                  'options'   => array(
                     ''          => t('--None--'),
                     'bg-theme'  => t('Background of theme'), 
                     'bg-theme-second'  => t('Background theme second'),
                     'bg-black'  => t('Background Black')
                  ) 
               ),
               array(
                  'id'        => 'animate',
                  'type'      => 'select',
                  'title'     => t('Animation'),
                  'desc'      => t('Entrance animation'),
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
               array(
                  'id'     => 'el_class',
                  'type'      => 'text',
                  'title'  => t('Extra class name'),
                  'desc'      => t('Style particular content element differently - add a class name and refer to it in custom CSS.'),
               ),

            ),                                       
         );
         return $fields;
      }

      public static function render_content( $attr = array(), $content = '' ){
         global $base_url;
         extract(gavias_merge_atts(array(
            'title'              => '',
            'sub_title'          => '',
            'content'            => '',
            'icon'               => '',
            'image'              => '',
            'layout_style'       => '',
            'box_background'     => '',
            'skin_text'          => '',            
            'el_class'           => '',
            'animate'            => '',
            'animate_delay'      => ''
         ), $attr));
         
         if($image) $image = $base_url . $image; 

         
         $classes = array();
         
         if($image) $classes[] = 'icon-image';
         if($el_class) $classes[] = $el_class;
         if($skin_text) $classes[] = $skin_text;
         
         if($box_background) $classes[] = '';
         if($layout_style) $classes[] = '';         
         $layout_class = "{$layout_style} {$box_background}";
         if($animate) $el_class .= ' wow ' . $animate;
         
         $_classes = count($classes) > 0 ? implode(' ', $classes) : ''; 
         ob_start();
         ?>
         
            <div class="widget gsc-box-icon <?php print $_classes ?> <?php print $layout_class ?>" <?php print gavias_content_builder_print_animate_wow('', $animate_delay) ?>>  
               <div class="box-inner">         
                  <?php if($icon){ ?>
                  <div class="box-icon">                     
                     <span class="icon <?php print $icon ?>"></span> 
                  </div>
                  <?php } ?>
                  <?php if($image){ ?>
                     <div class="box-image">
                        <img src="<?php print $image ?>" alt="<?php print strip_tags($title) ?>"/>  
                     </div>
                  <?php } ?>

                  <div class="box_content">
                     <?php if($sub_title){ ?>
                        <div class="sub-title"><?php print $sub_title; ?></div>
                     <?php } ?>
                     <?php if($title){ ?>
                        <div class="title"> <?php print $title; ?> </div>
                     <?php } ?>
                     <?php if($content){ ?>
                        <div class="desc"><?php print $content; ?></div>
                     <?php } ?>   
                  </div>
               </div>    
            </div> 
         

         <?php return ob_get_clean() ?>
       <?php
      }
   } 
endif;   

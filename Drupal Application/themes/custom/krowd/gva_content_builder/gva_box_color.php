<?php 

if(!class_exists('element_gva_box_color')):
   class element_gva_box_color{
      
      public function render_form(){
         $fields = array(
            'type'            => 'gsc_box_color',
            'title'           => t('Box color'),
            'size'            => 3,
            'fields' => array(
                array(
                  'id'        => 'title',
                  'type'      => 'text',
                  'title'     => 'Title for box',
                  'admin'     => true
               ),
               array(
                  'id'        => 'content',
                  'type'      => 'textarea',
                  'title'     => t('Content for box'),
               ),
               array(
                  'id'        => 'icon',
                  'type'      => 'text',
                  'title'     => t('Icon class'),
                  'std'       => '',
                  'desc'     => t('Use class icon font <a target="_blank" href="http://fontawesome.io/icons/">Icon Awesome</a>'),
               ),
               array(
                  'id'        => 'link',
                  'type'      => 'text',
                  'title'     => t('Link'),
               ),
               array(
                  'id'        => 'text_link',
                  'type'      => 'text',
                  'title'     => t('Text Link'),
                  'std'       => t('Read more')
               ),
               array(
                  'id'        => 'color',
                  'type'      => 'text',
                  'title'     => t('Background color'),
                  'desc'      => t('Background color fox box. e.g: #ccc')
               ),
               array(
                  'id'        => 'text_style',
                  'type'      => 'select',
                  'title'     => t('Text Style'),
                  'options'   => array( 'white' => t('White'), 'dark' => t('Dark') ),
               ),
               array(
                  'id'        => 'target',
                  'type'      => 'select',
                  'title'     => t('Open in new window'),
                  'desc'      => t('Adds a target="_blank" attribute to the link'),
                  'options'   => array( 0 => 'No', 1 => 'Yes' ),
               ),
               array(
                  'id'        => 'el_class',
                  'type'      => 'text',
                  'title'     => t('Extra class name'),
                  'desc'      => t('Style particular content element differently - add a class name and refer to it in custom CSS.'),
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
               )
            ),                                     
         );
         return $fields;
      }

      public static function render_content( $attr = array(), $content = null ){
         global $base_url;
         extract(gavias_merge_atts(array(
            'icon'                  => '',
            'title'                 => '',
            'content'               => '',
            'link'                  => '',
            'text_link'             => 'Read more',
            'color'                 => '',
            'text_style'            => 'white',
            'target'                => '',
            'el_class'              => '',
            'animate'               => '',
            'animate_delay'         => '',
         ), $attr));

         // target
         if( $target ){
            $target = 'target="_blank"';
         } else {
            $target = false;
         }
         $el_class .= ' text-' . $text_style;
         $style_bg = '';
         $style_color = '';
         if($color){
            $style_bg = "style=\"background-color: {$color};\"";
            $style_color = "style=\"color: {$color};\"";
         }
         if($animate) $el_class .= ' wow ' . $animate;
         ob_start();
         ?>
         <div class="widget gsc-box-color clearfix <?php print $el_class; ?>" <?php print gavias_content_builder_print_animate_wow('', $animate_delay) ?>>
            <div class="box-content" <?php print $style_bg ?>>
               <?php if($icon){ ?><div class="icon"><span class="<?php print $icon ?>" <?php print $style_color ?>></span></div> <?php } ?>
               <div class="widget-title box-title"><?php print $title ?></div>
               <div class="content"><?php print $content ?>
               <a class="link" <?php print $style_bg ?> <?php if($link) print 'href="'. $link .'"' ?> <?php print $target ?>><span class="text"><?php print $text_link ?></span><span class="background"></span></a>
               </div>
            </div>   
         </div>
         <?php return ob_get_clean() ?>
         <?php
      }

   }
endif;   





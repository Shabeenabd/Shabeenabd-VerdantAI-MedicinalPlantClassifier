<?php 
if(!class_exists('element_gva_call_to_action')):
   class element_gva_call_to_action{
      public function render_form(){
         $fields = array(
            'type' => 'gsc_call_to_action',
            'title' => t('Call to Action'),
            'size' => 12,
            
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
                  'desc'      => t('HTML tags allowed.'),
               ),
               array(
                  'id'        => 'link',
                  'type'      => 'text',
                  'title'     => t('Link'),
               ),
               array(
                  'id'        => 'button_title',
                  'type'      => 'text',
                  'title'     => t('Button Title'),
                  'desc'      => t('Leave this field blank if you want Call to Action with Big Icon'),
               ),
               array(
                  'id'           => 'button_align',
                  'type'         => 'select',
                  'title'        => 'Style',
                  'options'      => array(
                     ' '              => t('Default'),
                     'button-right'             => t('Button Right'),
                     'button-bottom v1'            => t('Button Bottom I'),
                     'button-bottom v2'         => t('Button Bottom II')
                  )
               ),
               array(
                  'id'        => 'width',
                  'type'      => 'text',
                  'title'     => t('Max width for content'),
                  'desc'      => 'e.g 660px'
               ),
               array(
                  'id'        => 'style_text',
                  'type'      => 'select',
                  'title'     => 'Skin Text for box',
                  'options'   => array(
                        'text-light'  => 'Text light',
                        'text-dark'   => 'Text dark',
                  ),
                  'std'       => 'text-dark'
               ),
               array(
                  'id'        => 'style_button',
                  'type'      => 'select',
                  'title'     => 'Style button',
                  'options'   => array(
                        'btn-theme'          => 'Button default of theme',
                        'btn-theme-second'          => 'Button Second of theme',
                        'btn-white'          => 'Button white',
                        'btn-black'          => 'Button Black' 
                  ),
                  'std'       => 'text-dark'
               ),
               array(
                  'id'        => 'target',
                  'type'      => 'select',
                  'title'     => t('Open in new window'),
                  'desc'      => t('Adds a target="_blank" attribute to the link'),
                  'options'   => array( 'off' => 'Off', 'on' => 'On' ),
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
               )
            )                                       
         );
      return $fields;
      }

      function render_content( $attr = array(), $content = '' ){
         extract(gavias_merge_atts(array(
            'title'           => '',
            'content'         => '',
            'link'            => '',
            'button_title'    => '',
            'button_align'    => '',
            'width'           => '',
            'style_button'    => 'btn-theme',
            'target'          => '',
            'el_class'        => '',
            'animate'         => '',
            'animate_delay'   => '',
            'style_text'      => 'text-dark',
         ), $attr));
         
         // target
         if( $target =='on' ){
            $target = 'target="_blank"';
         } else {
            $target = false;
         }
         
         $class = array();
         $class[] = $el_class;
         $class[] = $button_align;
         $class[] = $style_text;
        
         $width_style =  !empty( $width ) ? "style=\"max-width: {$width}\"" : '';
         if($animate) $class[] = ' wow ' . $animate;
         
          ob_start();
         ?>
         <div class="widget gsc-call-to-action <?php print implode(' ', $class) ?>" <?php print gavias_content_builder_print_animate_wow('', $animate_delay) ?>>
            <div class="content-inner clearfix" <?php print $width_style ?>>
               <div class="content">
                  <h2 class="title"><span><?php print $title; ?></span></h2>
                  <div class="desc"><?php print $content; ?></div>
               </div>
               <?php if($link){?>
               <div class="button-action">
                  <a href="<?php print $link ?>" class="<?php print $style_button ?>" <?php print $target ?>>
                     <span><?php print $button_title ?></span>
                  </a>   
               </div>
               <?php } ?>
            </div>
         </div>
         <?php return ob_get_clean() ?>
      <?php
      }
   }
endif;   




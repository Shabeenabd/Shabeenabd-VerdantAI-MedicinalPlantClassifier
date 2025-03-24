<?php 

if(!class_exists('element_gva_video_box')):
   class element_gva_video_box{

      public function render_form(){
         $fields = array(
            'type' => 'gsc_video_box',
            'title' => ('Video Box'), 
            'fields' => array(
               array(
                  'id'        => 'title',
                  'type'      => 'text',
                  'title'     => t('Title'),
                  'admin'     => true
               ),
               array(
                  'id'        => 'content',
                  'type'      => 'text',
                  'title'     => t('Data Url'),
                  'desc'      => t('example: https://www.youtube.com/watch?v=4g7zRxRN1Xk'),
               ),
               array(
                  'id'        => 'image',
                  'type'      => 'upload',
                  'title'     => t('Image Preview'),
               ),
               array(
                  'id'           => 'style',
                  'type'         => 'select',
                  'title'        => 'Style',
                  'options'      => array(
                     'style-1'         => t('Style 1'),
                     'style-2'         => t('Style 2'),
                     'style-3'         => t('Style 3')
                  )
               ),
              
               array(
                  'id'        => 'animate',
                  'type'      => 'select',
                  'title'     => t('Animation'),
                  'desc'      => t('Entrance animation for element'),
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
                  'id'        => 'el_class',
                  'type'      => 'text',
                  'title'     => t('Extra class name'),
                  'desc'      => t('Style particular content element differently - add a class name and refer to it in custom CSS.'),
               ),
            ),                                       
         );
         return $fields;
      }

      public static function render_content( $attr = array(), $content = '' ){
         global $base_url;
         extract(gavias_merge_atts(array(
            'title'           => '',
            'content'         => '',
            'image'           => '',
            'style'           => 'style-1',
            'sub_title'       => '',
            'link'            => '',
            'text_link'       => '',
            'animate'         => '',
            'animate_delay'   => '',
            'el_class'        => '',
         ), $attr));

         $_id = gavias_content_builder_makeid();
         if($image){
            $image = $base_url .$image; 
         }
         if($animate) $el_class .= ' wow ' . $animate;
         
         ob_start();
         ?>
      
      <div class="widget gsc-video-box <?php print $el_class;?> <?php print $style;?> <?php if($image) print ' bg-image' ?> clearfix" <?php print gavias_content_builder_print_animate_wow('', $animate_delay) ?>>
         <div class="video-inner">
            <?php if($image) { ?>
               <div class="image">
                  <img src="<?php print $image ?>"/>
               </div>
            <?php } ?>
            <div class="video-body">
               <a class="popup-video gsc-video-link" href="<?php print $content ?>">
                  <i class="fa fa-play"></i>
               </a>
            </div> 
         </div>     
      </div> 
      

            
      <?php return ob_get_clean() ?>
       <?php
      }

   }
endif;   





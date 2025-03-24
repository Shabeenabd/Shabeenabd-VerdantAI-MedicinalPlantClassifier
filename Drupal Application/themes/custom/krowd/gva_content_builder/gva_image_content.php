<?php 
if(!class_exists('element_gva_image_content')):
   class element_gva_image_content{
      public function render_form(){
         return array(
           'type'          => 'gsc_image_content',
            'title'        => t('Image content'),
            'fields' => array(
            
               array(
                  'id'        => 'title',
                  'type'      => 'text',
                  'title'     => t('Title'),
                  'admin'     => true
               ),
               array(
                  'id'        => 'icon',
                  'type'      => 'text',
                  'title'     => t('Icon'),
               ),
               array(
                  'id'        => 'background_1',
                  'type'      => 'upload',
                  'title'     => t('Images 1 Skin V1 & V2 & V3')
               ),
               array(
                  'id'        => 'background_2',
                  'type'      => 'upload',
                  'title'     => t('Images 2 Skin V1')
               ),
               array(
                  'id'        => 'color',
                  'type'      => 'text',
                  'title'     => t('Background color (use for skin #2)'),
                  'desc'      => t('Background color fox box. e.g: #ccc')
               ),
               array(
                  'id'        => 'content',
                  'type'      => 'textarea',
                  'title'     => t('Content'),
                  'desc'      => t('Some HTML tags allowed'),
               ),
               array(
                  'id'        => 'link',
                  'type'      => 'text',
                  'title'     => t('Link'),
                  'default'   => '#',
               ),
               array(
                  'id'        => 'text_link',
                  'type'      => 'text',
                  'title'     => t('Text Link'),
               ),
               array(
                  'id'        => 'target',
                  'type'      => 'select',
                  'title'     => t('Open in new window'),
                  'desc'      => t('Adds a target="_blank" attribute to the link'),
                  'options'   => array( 'off' => 'No', 'on' => 'Yes' ),
                  'std'       => 'on'
               ),
               array(
                  'id'        => 'skin',
                  'type'      => 'select',
                  'title'     => t('Skin'),
                  'options'   => array( 
                     'skin-v1' => t('Skin I'), 
                     'skin-v2' => t('Skin II'),
                     'skin-v3' => t('Skin III')
                  ),
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
               ), 
            ),                                     
         );
      }

      public static function render_content( $attr = array(), $content = '' ){
         global $base_url;
         extract(gavias_merge_atts(array(
            'title'              => '',
            'icon'               => '',
            'background_1'       => '',
            'background_2'       => '', 
            'color'              => '',
            'content'            => '',
            'link'               => '#',
            'text_link'          => 'Read more',
            'target'             => '',
            'skin'               => 'skin-v1',
            'el_class'           => '',
            'animate'            => '',
            'animate_delay'      => '',
         ), $attr));

         // target
         if( $target =='on' ){
            $target = 'target="_blank"';
         } else {
            $target = false;
         }
         $background_1 = $base_url . $background_1; 
         $background_2 = $base_url . $background_2; 

         if($skin) $el_class .= ' ' . $skin;
         if($animate) $el_class .= ' wow ' . $animate;
         
         ob_start();
         ?>

         <?php if($skin == 'skin-v1'){ ?>
            <div class="gsc-image-content <?php print $el_class; ?>" <?php print gavias_content_builder_print_animate_wow('', $animate_delay) ?>>
               <div class="line-color"></div>
               <div class="image">
                  <?php if($link && $background_1){ ?>
                     <a <?php print $target ?> href="<?php print $link ?>"><?php } ?>
                        <img src="<?php print $background_1 ?>" alt="<?php print $title ?>" />
                        <?php if($link){ ?>
                     </a>
                  <?php } ?>
               </div>
               <div class="image-second">
                  <?php if($link && $background_2){ ?>
                     <a <?php print $target ?> href="<?php print $link ?>"><?php } ?>
                        <img src="<?php print $background_2 ?>" alt="<?php print $title ?>" />
                        <?php if($link){ ?>
                     </a>
                  <?php } ?>
               </div>  
            </div>
         <?php } ?>   

         <?php if($skin == 'skin-v2'){ ?>

            <div class="gsc-image-content <?php print $el_class; ?>" <?php print gavias_content_builder_print_animate_wow('', $animate_delay) ?>>               
               <div class="image">
                  <?php if($link){ ?>
                     <a <?php print $target ?> href="<?php print $link ?>"><?php } ?>
                        <img src="<?php print $background_1 ?>" alt="<?php print $title ?>" />
                        <?php if($link){ ?>
                     </a>
                  <?php } ?>
               </div>
               <div class="content">
                  <?php if($title){ ?>
                     <h4 class="title">
                        <?php if($link){ ?>
                           <a <?php print $target ?> href="<?php print $link ?>">
                           <?php } ?>
                           <?php print $title ?>
                           <?php if($link){ ?>
                           </a><?php } ?>  
                     </h4>
                  <?php } ?>   
               </div> 
            </div>
         <?php } ?> 

         <?php if($skin == 'skin-v3'){ ?>

            <div class="gsc-image-content <?php print $el_class; ?>" <?php print gavias_content_builder_print_animate_wow('', $animate_delay) ?>>               
               <div class="image">
                  <?php if($link){ ?>
                     <a <?php print $target ?> href="<?php print $link ?>"><?php } ?>
                        <img src="<?php print $background_1 ?>" alt="<?php print $title ?>" />
                        <?php if($link){ ?>
                     </a>
                  <?php } ?>
               </div>
               <div class="content">
                  <?php if($title){ ?>
                     <h4 class="title">
                        <?php if($link){ ?>
                           <a <?php print $target ?> href="<?php print $link ?>">
                           <?php } ?>
                           <?php print $title ?>
                           <?php if($link){ ?>
                           </a><?php } ?>  
                     </h4>
                  <?php } ?>  
                  <?php if($content){ ?>
                     <div class="desc"><?php print $content ?></div>
                  <?php } ?>  
               </div> 
            </div>
         <?php } ?>

         <?php return ob_get_clean() ?>
        <?php            
      } 

   }
endif;   

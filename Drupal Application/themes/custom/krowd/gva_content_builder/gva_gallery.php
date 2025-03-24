<?php 

if(!class_exists('element_gva_gallery')):
   class element_gva_gallery{

      public function render_form(){
         $fields = array(
            'type' => 'gsc_gallery',
            'title' => t('Gallery'),
            'size' => 3,
            'fields' => array(
               array(
                  'id'        => 'title',
                  'type'      => 'text',
                  'title'     => t('Title For Admin'),
                  'admin'     => true
               ),
               array(
                  'id'        => 'animate',
                  'type'      => 'select',
                  'title'     => ('Animation'),
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
               array(
                  'id'        => 'data_items',
                  'type'      => 'text',
                  'title'     => t('Number column'),
               ),
               array(
                 'id'        => 'style_layout',
                 'type'      => 'select',
                 'title'     => 'Style Layout',
                 'options'   => array(
                     'carousel'            => 'Carousel',
                     'grid-column-3'       => 'Grid Column 3'
                 )
               ),
               array(
                 'id'        => 'style_space',
                 'type'      => 'select',
                 'title'     => 'Style Space',
                 'options'   => array(
                     'default'                       => 'Default',
                     'padding-small'                 => 'Padding Small',
                     'remove-padding'                => 'Remove padding'
                 )
               ),
            ),                                     
         );

         for($i=1; $i<=10; $i++){
            $fields['fields'][] = array(
               'id'     => "info_{$i}",
               'type'   => 'info',
               'desc'   => "Information for item {$i}"
            );
            $fields['fields'][] = array(
               'id'        => "title_{$i}",
               'type'      => 'text',
               'title'     => t("Title {$i}")
            );
            $fields['fields'][] = array(
               'id'        => "image_{$i}",
               'type'      => 'upload',
               'title'     => t("Image {$i}")
            );
         }
         return $fields;
      }

      public static function render_content( $attr = array(), $content = '' ){
         global $base_url;
         $default = array(
            'title'           => '',
            'el_class'        => '',
            'animate'         => '',
            'animate_delay'   => '',
            'data_items'      => '',
            'style_space'     => '',
            'style_layout'    => ''
         );

         for($i=1; $i<=10; $i++){
            $default["title_{$i}"] = '';
            $default["image_{$i}"] = '';
         }

         extract(gavias_merge_atts($default, $attr));

         $_id = gavias_content_builder_makeid();
         if($animate) $el_class .= ' wow ' . $animate;
         
          ob_start();
         ?>
         <?php if($style_layout == 'carousel'){ ?>
         <div class="gsc-our-gallery <?php print $el_class ?> <?php print $style_space ?>" <?php print gavias_content_builder_print_animate_wow('', $animate_delay) ?>> 
            <div class="owl-carousel init-carousel-owl owl-loaded owl-drag" data-items="<?php print $data_items ?>" data-items_sm="3" data-items_xs="2" data-loop="1" data-speed="1000" data-auto_play="1" data-auto_play_speed="2000" data-auto_play_timeout="5000" data-auto_play_hover="1" data-navigation="1" data-rewind_nav="0" data-pagination="0" data-mouse_drag="1" data-touch_drag="1">
               <?php for($i=1; $i<=10; $i++){ ?>
                  <?php 
                     $title = "title_{$i}";
                     $image = "image_{$i}";
                  ?>
                  <?php if($$title || $$image){ ?>
                     <div class="item"><div class="content-inner">
                        <?php if($$title){ ?><div class="title"><?php print $$title ?></div><?php } ?>         
                        <?php if($$image){ ?>
                           <div class="image">
                              <a href="<?php echo ($base_url . $$image) ?>" data-rel="prettyPhoto[g_gal]">
                                 <span class="icon-expand"><i class="icon fi flaticon-plus"></i></span>
                                 <img src="<?php echo ($base_url . $$image) ?>" />
                              </a>
                           </div>
                        <?php } ?>
                     </div></div>
                  <?php } ?>    
               <?php } ?>
            </div> 
         </div> 
         <?php } ?>

         <?php if($style_layout == 'grid-column-3'){ ?>
         <div class="gsc-our-gallery <?php print $el_class ?> <?php print $style_space ?>  <?php print $style_layout ?>" <?php print gavias_content_builder_print_animate_wow('', $animate_delay) ?>> 
            <div class="row">
               <?php for($i=1; $i<=10; $i++){ ?>
               <?php 
                  $title = "title_{$i}";
                  $image = "image_{$i}";
               ?>
               <?php if($$title || $$image){ ?>
                  <div class="item col-xl-4 col-lg-4 col-md-4 col-sm-12 col-xs-12"><div class="content-inner">
                     <?php if($$title){ ?><div class="title"><?php print $$title ?></div><?php } ?>         
                     <?php if($$image){ ?>
                        <div class="image">
                           <a href="<?php echo ($base_url . $$image) ?>" data-rel="prettyPhoto[g_gal]">
                              <span class="icon-expand"><i class="icon fi flaticon-plus"></i></span>
                              <img src="<?php echo ($base_url . $$image) ?>" />
                           </a>
                        </div>
                     <?php } ?>
                  </div></div>
               <?php } ?>    
            <?php } ?>
            </div>
         </div>
         <?php } ?>

         <?php return ob_get_clean();
      }

   }
 endif;  




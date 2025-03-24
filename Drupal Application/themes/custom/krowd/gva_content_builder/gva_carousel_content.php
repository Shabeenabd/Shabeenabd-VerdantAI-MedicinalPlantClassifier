<?php 
if(!class_exists('element_gva_carousel_content')):
   class element_gva_carousel_content{
      public function render_form(){
         $fields = array(
            'type' => 'gsc_carousel_content',
            'title' => t('Carousel Content'),
            'size' => 3,
            'fields' => array(
               array(
                  'id'        => 'title',
                  'type'      => 'text',
                  'title'     => t('Title For Admin'),
                  'admin'     => true
               ),
               array(
                  'id'        => 'bg_color',
                  'type'      => 'text',
                  'title'     => t('Background color'),
                  'desc'      => t('Background color fox box. e.g: #ccc, default background of theme')
               ),
               array(
                  'id'        => 'style',
                  'type'      => 'select',
                  'title'     => 'Style',
                  'options'   => array( 
                     'style-1'   => 'Style #1 (color white)', 
                     'style-2'   => 'Style #2 (color dark)',
                     'style-3'   => 'Style #3 (color white, parallax)'
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

         for($i=1; $i<=8; $i++){
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
               'id'           => "content_{$i}",
               'type'         => 'textarea',
               'title'        => t("Content {$i}"),
               'desc'         => t('Shortcodes and HTML tags allowed.'),
               'shortcodes'   => 'on'
            );
            $fields['fields'][] = array(
               'id'        => "link_{$i}",
               'type'      => 'text',
               'title'     => t("Link {$i}")
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
            'bg_color'        => '',
            'style'           => 'style-1',
            'el_class'        => '',
            'animate'         => '',
            'animate_delay'   => '',
         );

         for($i=1; $i<=10; $i++){
            $default["title_{$i}"] = '';
            $default["content_{$i}"] = '';
            $default["link_{$i}"] = '';
            $default["image_{$i}"] = '';
         }

         extract(gavias_merge_atts($default, $attr));

         $el_class .= ' ' . $style;
         $_id = gavias_content_builder_makeid();
         $bg_style = '';
         if($bg_color){
            $bg_style = "style=\"background-color: {$bg_color};\"";
         }
         if($animate) $el_class .= ' wow ' . $animate;

         ob_start();
         ?>
         
         <?php if($style == 'style-3'){ ?>

            <div class="gsc-carousel-content <?php echo $el_class ?>" <?php print gavias_content_builder_print_animate_wow('', $animate_delay) ?>> 
               <div class="owl-carousel init-carousel-owl"data-items="1" data-items_lg="1" data-items_md="1" data-items_sm="1" data-items_xs="1" data-loop="1" data-speed="500" data-auto_play="0" data-auto_play_speed="0" data-auto_play_timeout="0" data-auto_play_hover="1" data-navigation="0" data-rewind_nav="0" data-pagination="1" data-mouse_drag="1" data-touch_drag="1">
                  <?php for($i=1; $i<=8; $i++){ ?>
                     <?php 
                        $title = "title_{$i}";
                        $content = "content_{$i}";
                        $link = "link_{$i}";
                        $image = "image_{$i}";
                     ?>
                     <?php if($$title){ ?>
                        <div class="item">
                           <div class="item-content">
                              <?php if($$image){?>
                                <div class="image"><img src="<?php echo ($base_url . $$image) ?>" /></div>
                              <?php }  ?>
                              <div class="content-box skrollable skrollable-between" data-bottom-top="top: -10%;" data-top-bottom="top: 30%;" <?php print $bg_style ?>>
                                 <div class="content-inner">
                                    <?php if($$title){ ?>
                                       <div class="title"><a href="<?php print $$link ?>"><?php print $$title ?></a></div>
                                    <?php } ?>
                                    <?php if($$content){ echo '<div class="desc">' . $$content . '</div>'; } ?>
                                    <?php if($$link){ ?>
                                       <div class="action margin-top-30"><a class="btn btn-theme" href="<?php print $$link ?>"><?php print t('Read more') ?></a></div>
                                    <?php } ?>
                                 </div>
                              </div>
                           </div>
                        </div>
                     <?php } ?>    
                  <?php } ?>
               </div>
            </div> 

         <?php }else{ ?>

         <div class="gsc-carousel-content <?php echo $el_class ?>"> 
            <div class="owl-carousel init-carousel-owl"data-items="1" data-items_lg="1" data-items_md="1" data-items_sm="1" data-items_xs="1" data-loop="1" data-speed="500" data-auto_play="0" data-auto_play_speed="0" data-auto_play_timeout="0" data-auto_play_hover="1" data-navigation="1" data-rewind_nav="0" data-pagination="0" data-mouse_drag="1" data-touch_drag="1">
               <?php for($i=1; $i<=8; $i++){ ?>
                  <?php 
                     $title = "title_{$i}";
                     $content = "content_{$i}";
                     $link = "link_{$i}";
                     $image = "image_{$i}";
                  ?>
                  <?php if($$title){ ?>
                     <div class="item">
                        <div class="item-content">
                           <?php if($$image){?>
                             <div class="bg-full" style="background-image:url('<?php echo ($base_url . $$image) ?>') "></div>
                           <?php }  ?>
                           <div class="content-box" <?php print $bg_style ?>>
                              <div class="content-inner">
                                 <?php if($$title){ ?>
                                    <div class="title"><a href="<?php print $$link ?>"><?php print $$title ?></a></div>
                                 <?php } ?>
                                 <?php if($$content){ echo '<div class="desc">' . $$content . '</div>'; } ?>
                                 <?php if($$link){ ?>
                                    <div class="action margin-top-30"><a class="btn btn-theme" href="<?php print $$link ?>"><?php print t('Read more') ?></a></div>
                                 <?php } ?>
                              </div>
                           </div>
                        </div>
                     </div>
                  <?php } ?>    
               <?php } ?>
            </div>   
         </div> 
         <?php } ?>  

         <?php return ob_get_clean();
      }
   }
 endif;  




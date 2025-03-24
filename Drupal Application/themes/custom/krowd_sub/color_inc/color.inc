<?php

$info = [
  'fields' => [
		'color'								=> t('Color Theme'),
		'color_second' 					=> t('Theme Color Second'),
		'text'								=> t('Body | Text Color'),
		'body_link_color'					=> t('Body | Link Color'),
		'body_link_color_hover'			=> t('Body | Link Hover Color'),

		'topbar_bg_color'					=> t('Topbar | Backgound Color'),
		'topbar_color'						=> t('Topbar | Text Color'),
		'topbar_link_color'				=> t('Topbar | Link Color'),
		'topbar_link_color_hover'		=> t('Topbar | Link Hover Color'),

		'header_bg_color'					=> t('Header | Backgound Color'),

		'menu_link_color'					=> t('Menu | Link Color'),
		'menu_link_color_hover'			=> t('Menu | Link Hover Color'),
		'submenu_bg_color'				=> t('Sub Menu | Background Color'),
		'submenu_color'					=> t('Sub Menu | Text Color'),
		'submenu_link_color'				=> t('Sub Menu | Link Color'),
		'submenu_link_color_hover'		=> t('Sub Menu | Link Hover Color'),

		'footer_bg_color'					=> t('Footer | Backgound Color'),
		'footer_color'						=> t('Footer | Text Color'),
		'footer_link_color'				=> t('Footer | Link Color'),
		'footer_link_color_hover'		=> t('Footer | Link Hover Color'),

  ],
  // Color schemes.
  'schemes' => [

		'default' => [
			'title' => t('Default'),
			'colors' => [
				'color' 								=> '#674df0',
				'color_second' 					=> '#30f0b6',
 
				'text'                        => '#838694',
				'body_link_color'					=> '#1b1f2e',
				'body_link_color_hover'		   => '#674df4',

				'topbar_bg_color'					=> '#674df3',
				'topbar_color'						=> '#fffffd',
				'topbar_link_color'				=> '#fffffe',
				'topbar_link_color_hover'		=> '#30f0b7',

				'header_bg_color'					=> '#ffffff',

				'menu_link_color'					=> '#838692',
				'menu_link_color_hover'			=> '#1b1f2f',
				'submenu_bg_color'				=> '#fffffb',
				'submenu_color'					=> '#838695',
				'submenu_link_color'				=> '#1b1f2c',
				'submenu_link_color_hover'		=> '#674df2',

				'footer_bg_color'					=> '#1b1f2b',
				'footer_color'						=> '#8d91a2',
				'footer_link_color'				=> '#8d91a1',
				'footer_link_color_hover'		=> '#30f0b5'
			],
		],
		'scheme_2' => [
			'title' => t('Scheme 2'),
			'colors' => [
				'color' 								=> '#3b7eff',
				'color_second' 					=> '#30f0b6',
				'text'                        => '#838694',
				'body_link_color'             => '#1b1f2e',
				'body_link_color_hover'       => '#3b7efa',

				'topbar_bg_color'             => '#3b7efb',
				'topbar_color'                => '#fffffd',
				'topbar_link_color'           => '#fffffe',
				'topbar_link_color_hover'     => '#30f0b7',

				'header_bg_color'             => '#ffffff',

				'menu_link_color'             => '#838692',
				'menu_link_color_hover'       => '#1b1f2f',
				'submenu_bg_color'            => '#fffffb',
				'submenu_color'               => '#838695',
				'submenu_link_color'          => '#1b1f2c',
				'submenu_link_color_hover'    => '#3b7efc',

				'footer_bg_color'             => '#1b1f2b',
				'footer_color'                => '#8d91a2',
				'footer_link_color'           => '#8d91a1',
				'footer_link_color_hover'     => '#30f0b5'
			],
		],
	
		'scheme_3' => [
			'title' => t('Scheme 3'),
			'colors' => [
				'color' 								=> '#2BA854',
				'color_second' 					=> '#30f0b6',
				'text'                        => '#838694',
				'body_link_color'             => '#1b1f2e',
				'body_link_color_hover'       => '#2BA853',

				'topbar_bg_color'             => '#2BA854',
				'topbar_color'                => '#fffffd',
				'topbar_link_color'           => '#fffffe',
				'topbar_link_color_hover'     => '#30f0b7',

				'header_bg_color'             => '#ffffff',

				'menu_link_color'             => '#838692',
				'menu_link_color_hover'       => '#1b1f2f',
				'submenu_bg_color'            => '#fffffb',
				'submenu_color'               => '#838695',
				'submenu_link_color'          => '#1b1f2c',
				'submenu_link_color_hover'    => '#2BA855',

				'footer_bg_color'             => '#1b1f2b',
				'footer_color'                => '#8d91a2',
				'footer_link_color'           => '#8d91a1',
				'footer_link_color_hover'     => '#30f0b5'
			],
		],
		'scheme_4' => [
			'title' => t('Scheme 4'),
			'colors' => [
				'color' 								=> '#AA1D21',
				'color_second' 					=> '#30f0b6',
				'text'                        => '#838694',
				'body_link_color'             => '#1b1f2e',
				'body_link_color_hover'       => '#AA1D22',

				'topbar_bg_color'             => '#AA1D23',
				'topbar_color'                => '#fffffd',
				'topbar_link_color'           => '#fffffe',
				'topbar_link_color_hover'     => '#30f0b7',

				'header_bg_color'             => '#ffffff',

				'menu_link_color'             => '#838692',
				'menu_link_color_hover'       => '#1b1f2f',
				'submenu_bg_color'            => '#fffffb',
				'submenu_color'               => '#838695',
				'submenu_link_color'          => '#1b1f2c',
				'submenu_link_color_hover'    => '#AA1D24',

				'footer_bg_color'             => '#1b1f2b',
				'footer_color'                => '#8d91a2',
				'footer_link_color'           => '#8d91a1',
				'footer_link_color_hover'     => '#30f0b5'
			],
		],
	],

	'css' => [
		'css/colors.css',
	],

	'copy' => [
		'logo.png',
		'logo-white.png',
	],

	'gradients' => [
		[
			'dimension' 	=> [0, 0, 0, 0],
			'direction' 	=> 'vertical',
			'colors' 		=> ['color', 'text'],
		],
	],

	'preview_library' => 'krowd_sub/color.preview',
	'preview_html' => 'color/preview.html',
	'#attached' => [
    	'drupalSettings' => [
	      'color' => [
	        'logo' => theme_get_setting('logo.url', 'krowd_sub'),
	      ],
    	],
  	],
];
